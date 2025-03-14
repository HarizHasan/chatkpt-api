import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

#  Import functions from chatkpt.py
from chatkpt import (
    download_files_from_gcs,
    process_documents,
    text_to_embed,
    bigquery_insert,
    query_similar_text,
    rag_query
)

app = FastAPI(title="ChatKPT API")

#  Store processed documents/embeddings
app.state.documents = None
app.state.document_embeddings = None

async def run_processing_pipeline():
    """Downloads and processes documents and inserts embeddings into BigQuery"""
    await asyncio.sleep(1)  # Prevents async issues

    download_files_from_gcs()
    documents = process_documents()
    if not documents:
        raise RuntimeError("❌ No documents found! Ensure Cloud Storage contains valid files.")
    document_embeddings = text_to_embed(documents)
    bigquery_insert(document_embeddings)

    #  Store data globally in app.state
    app.state.documents = documents
    app.state.document_embeddings = document_embeddings

    print("✅ Processing pipeline completed.")

@app.on_event("startup")
async def startup_event():
    """Run the processing pipeline once at startup"""
    print("Starting document processing pipeline on startup...")

    # Run in background to prevent Cloud Run timeout
    loop = asyncio.get_event_loop()
    loop.create_task(run_processing_pipeline())

    print("✅ Processing pipeline started successfully!")

@app.post("/reprocess-documents")
async def reprocess_documents():
    """Manually trigger the document processing pipeline without restarting the app"""
    if app.state.documents is not None:
        return {"message": "❗ Documents are already processed. Updating documents..."}
    
    print("Starting document processing pipeline")

    loop = asyncio.get_event_loop()
    loop.create_task(run_processing_pipeline())

    return {"message": "✅ Processing pipeline started successfully!"}

#  Pydantic model for query requests
class QueryRequest(BaseModel):
    user_query: str

@app.post("/query")
async def query_docs(request: QueryRequest):
    """Query processed documents. Uses pre-loaded documents/embeddings stored in app.state"""
    try:
        if app.state.documents is None:
            raise RuntimeError("❌ Documents have not been processed yet!")
        results = query_similar_text(request.user_query)
        response_text = rag_query(results, request.user_query)
        return {"results": results, "rag_response": response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
