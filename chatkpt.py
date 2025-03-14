import google.generativeai as genai
from google.cloud import aiplatform, bigquery, storage
from vertexai.language_models import TextEmbeddingModel
import json
import numpy as np
import pdfplumber
from docx import Document
import os
import glob
import dotenv
from dotenv import load_dotenv

load_dotenv()

# Set up Google Cloud project
PROJECT_ID = 'anaylytic-chatkpt' # Project ID here
aiplatform.init(project=PROJECT_ID, location='asia-southeast1')

# Set up BigQuery client
BIGQUERY_CLIENT = bigquery.Client()
TABLE_ID = "anaylytic-chatkpt.rag_bq.my_vectors" # Create table first
MAX_CHUNK_SIZE = 1000  # BigQuery limit safeguard (adjust if needed)

# Set up GCS client
STORAGE_CLIENT = storage.Client()
BUCKET_NAME = "chatkpt_dev_bucket"  # Bucket here
DESTINATION_FOLDER = "downloaded_docs"

def download_files_from_gcs():
    """Downloads all files from the given GCS bucket"""
    if not os.path.exists(DESTINATION_FOLDER):
        os.makedirs(DESTINATION_FOLDER)  # Ensure the local folder exists

    bucket = STORAGE_CLIENT.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs()  # List all files in the bucket

    for blob in blobs:
        file_path = os.path.join(DESTINATION_FOLDER, blob.name)
        print(f"Downloading {blob.name} to {file_path}...")
        blob.download_to_filename(file_path)

    print("✅ All files downloaded!")

def extract_text_from_txt(file_path):
    """Extracts text from a .txt file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF (.pdf) file"""
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text
   
def extract_text_from_docx(file_path):
    """Extracts text from a Word (.docx) file."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def process_documents():
    """Processes all documents in the folder and extracts text"""
    all_texts = []

    for file_path in glob.glob(f"{DESTINATION_FOLDER}/*"):
        file_name = os.path.basename(file_path)

        # Determine file type and extract text
        if file_path.endswith(".txt"):
            text = extract_text_from_txt(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print(f"⚠️ Skipping unsupported file: {file_name}")
            continue

        all_texts.append((file_name, text))
        print(f"✅ Processed {file_name}")

    return all_texts

def generate_embedding(text):
    """Generates embeddings based on provided text"""
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    embedding = model.get_embeddings([text])[0]
    return embedding.values  # Returns a list of float values

def text_to_embed(documents):
    """Converts all processed text to embeddings"""
    document_embeddings = []
    for file_name, text in documents:
        embedding = generate_embedding(text)
        document_embeddings.append((file_name, text, embedding))
    print(f"✅ Generated embeddings for {len(document_embeddings)} documents")
    print('Generated embedding:', embedding[:5], '...')
    return document_embeddings

def split_text(text, MAX_CHUNK_SIZE):
    """Splits text into chunks of `chunk_size` characters to bypass BigQuery row limit."""
    return [text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text), MAX_CHUNK_SIZE)]

def bigquery_insert(document_embeddings):
    """Inserts all embeddiings into BigQuery"""
    rows_to_insert = []
    for file_name, text, embedding in document_embeddings:
        text_chunks = split_text(text,MAX_CHUNK_SIZE)  # Split large text into smaller chunks

        for chunk_id, chunk in enumerate(text_chunks):
            rows_to_insert.append({
                "id": file_name,
                "chunk_id": chunk_id,
                "text": chunk,
                "embedding": embedding  # Store the same embedding for consistency
            })

    errors = BIGQUERY_CLIENT.insert_rows_json(TABLE_ID, rows_to_insert)

    if errors:
        print("❌ Error inserting into BigQuery:", errors)
    else:
        print(f"✅ Successfully stored {len(rows_to_insert)} text chunks in BigQuery")

def query_similar_text(user_query):
    """Queries BigQuery based on user input"""
    query_embedding = generate_embedding(user_query)

    # Convert embedding list to a properly formatted BigQuery array
    embedding_str = ', '.join(map(str, query_embedding))

    query = f"""
    WITH query_embedding AS (
      SELECT ARRAY<FLOAT64>[{embedding_str}] AS embedding
    )
    SELECT 
        id, 
        text, 
        (SELECT SUM(x * y) 
         FROM UNNEST(v.embedding) AS x, UNNEST(q.embedding) AS y) AS similarity
    FROM `anaylytic-chatkpt.rag_bq.my_vectors` AS v, query_embedding AS q
    ORDER BY similarity DESC
    LIMIT 5;
    """

    query_job = BIGQUERY_CLIENT.query(query)
    results = query_job.result()

    return [(row.text, row.similarity) for row in results]

def rag_query(retrieved_docs, user_query):
    """Queries Gemini to generate a response based on retrieved document chunks."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')

    context_text = '\n'.join([doc[0] for doc in retrieved_docs])
    response = model.generate_content([
        f'Context: {context_text}',
        f'User Question: {user_query}'
    ])

    return response.text

if __name__ == '__main__':
    # Runs only if name is 'main'
    download_files_from_gcs()  # Download and process documents
    documents = process_documents()
    document_embeddings = text_to_embed(documents)
    bigquery_insert(document_embeddings)
    user_query = input("Input query: ")  # Get user query, retrieve documents and provide summary via RAG based on user query, then return response
    retrieved_docs = query_similar_text(user_query)
    text=rag_query(retrieved_docs, user_query)
    print(text)