import os
import sys

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_postgres import PGVector


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

COLLECTION_NAME = os.getenv(
    "PG_VECTOR_COLLECTION_NAME",
    "pdf_chunks",
)

EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

CHAT_MODEL = os.getenv(
    "GOOGLE_CHAT_MODEL",
    "gemini-3.5-flash-lite",
)

EMBEDDING_DIMENSIONS = 768
TOP_K = 10

FALLBACK_RESPONSE = (
    "Não tenho informações necessárias para responder sua pergunta."
)


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def create_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIMENSIONS,
        task_type="RETRIEVAL_QUERY",
    )

    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embedding_length=EMBEDDING_DIMENSIONS,
        use_jsonb=True,
        pre_delete_collection=False,
    )


def search_documents(vector_store, question):
    return vector_store.similarity_search_with_score(
        query=question,
        k=TOP_K,
    )


def build_context(results):
    return "\n\n".join(
        (
            f"Documento {index}\n"
            f"Página: {document.metadata.get('page_label', 'Desconhecida')}\n"
            f"Conteúdo:\n{document.page_content}"
        )
        for index, (document, _) in enumerate(results, start=1)
    )


def create_search_chain():
    vector_store = create_vector_store()

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        max_retries=2,
    )

    def answer_question(question: str) -> str:
        question = question.strip()

        if not question:
            raise ValueError(
                "A pergunta não pode estar vazia."
            )

        results = search_documents(vector_store, question)

        if not results:
            return FALLBACK_RESPONSE

        context = build_context(results)

        prompt = PROMPT_TEMPLATE.format(
            contexto=context,
            pergunta=question,
        )

        response = llm.invoke(prompt)

        answer = response.text.strip()

        return answer or FALLBACK_RESPONSE

    return RunnableLambda(answer_question)


def search_prompt(question=None):
    chain = create_search_chain()

    if question is not None:
        return chain.invoke(question)

    return chain


def main():
    if len(sys.argv) < 2:
        print(
            'Uso: python src/search.py "sua pergunta"'
        )
        return

    question = " ".join(sys.argv[1:])

    print(f"Pergunta: {question}")
    print("Buscando documentos e gerando resposta...")

    answer = search_prompt(question)

    print(f"\nResposta: {answer}")


if __name__ == "__main__":
    main()
