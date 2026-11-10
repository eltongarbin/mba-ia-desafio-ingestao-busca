import os
import time
import uuid

import psycopg
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = os.getenv(
    "PG_VECTOR_COLLECTION_NAME",
    "pdf_chunks",
)

EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_DIMENSIONS = 768

BATCH_SIZE = 5
BATCH_DELAY_SECONDS = 10
MAX_RETRIES = 5


def create_chunk_id(chunk, index):
    identifier = (
        f"{COLLECTION_NAME}:"
        f"{PDF_PATH}:"
        f"{chunk.metadata.get('page')}:"
        f"{index}:"
        f"{chunk.page_content}"
    )

    return str(uuid.uuid5(uuid.NAMESPACE_URL, identifier))


def get_existing_ids(chunk_ids):
    connection_url = DATABASE_URL.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )

    with psycopg.connect(connection_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT embedding.id
                FROM langchain_pg_embedding AS embedding
                JOIN langchain_pg_collection AS collection
                    ON embedding.collection_id = collection.uuid
                WHERE collection.name = %s
                    AND embedding.id = ANY(%s::varchar[])
                """,
                (COLLECTION_NAME, chunk_ids),
            )

            return {row[0] for row in cursor.fetchall()}


def store_batch(vector_store, batch, batch_ids):
    for attempt in range(MAX_RETRIES + 1):
        try:
            vector_store.add_documents(
                documents=batch,
                ids=batch_ids,
            )

            return

        except Exception as error:
            error_message = str(error)

            is_rate_limit = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            if not is_rate_limit or attempt == MAX_RETRIES:
                raise

            delay = min(30 * (2 ** attempt), 180)

            print(
                f"Limite da API atingido. "
                f"Nova tentativa em {delay} segundos..."
            )

            time.sleep(delay)


def ingest_pdf():
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "A variável GOOGLE_API_KEY não está configurada."
        )

    if not DATABASE_URL:
        raise ValueError(
            "A variável DATABASE_URL não está configurada."
        )

    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado: {PDF_PATH}"
        )

    print("\nCarregando PDF...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Total de páginas: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total de chunks gerados: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "Nenhum fragmento de texto foi gerado."
        )

    print("\nInicializando embeddings do Gemini...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIMENSIONS,
        task_type="RETRIEVAL_DOCUMENT",
    )

    print("Conectando ao PostgreSQL...")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embedding_length=EMBEDDING_DIMENSIONS,
        use_jsonb=True,
        pre_delete_collection=False,
    )

    chunk_ids = [
        create_chunk_id(chunk, index)
        for index, chunk in enumerate(chunks)
    ]

    existing_ids = get_existing_ids(chunk_ids)

    pending_chunks = [
        (chunk, chunk_id)
        for chunk, chunk_id in zip(chunks, chunk_ids)
        if chunk_id not in existing_ids
    ]

    print(f"\nChunks já armazenados: {len(existing_ids)}")
    print(f"Chunks pendentes: {len(pending_chunks)}")

    for start in range(0, len(pending_chunks), BATCH_SIZE):
        batch_items = pending_chunks[start:start + BATCH_SIZE]

        batch = [chunk for chunk, _ in batch_items]
        batch_ids = [chunk_id for _, chunk_id in batch_items]

        store_batch(vector_store, batch, batch_ids)

        processed = start + len(batch)

        print(
            f"Armazenados {processed}/{len(pending_chunks)} "
            f"chunks pendentes"
        )

        if processed < len(pending_chunks):
            time.sleep(BATCH_DELAY_SECONDS)

    print("\nIngestão concluída com sucesso!")

    return len(chunks)


if __name__ == "__main__":
    ingest_pdf()
