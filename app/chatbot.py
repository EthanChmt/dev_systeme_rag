import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


load_dotenv()

INDEX_PATH = Path("vectorstore/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def format_context(documents):
    context_parts = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata

        context_parts.append(
            f"""
Événement {index}
Titre : {metadata.get("title", "")}
Date de début : {metadata.get("begin", "")}
Date de fin : {metadata.get("end", "")}
Lieu : {metadata.get("location_name", "")}
Ville : {metadata.get("city", "")}
Adresse : {metadata.get("address", "")}
Description :
{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


def ask_chatbot(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("La question ne peut pas être vide.")

    vectorstore = load_vectorstore()

    documents = vectorstore.similarity_search(
        question,
        k=4,
    )

    context = format_context(documents)

    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        temperature=0,
        max_retries=2,
    )

    messages = [
        (
            "system",
            """
Tu es un assistant spécialisé dans les recommandations culturelles.

Réponds uniquement à partir des événements présents dans le contexte fourni.

Règles :
- N'invente aucun événement, lieu, date ou tarif.
- Si le contexte ne permet pas de répondre, indique clairement que tu ne disposes pas d'information suffisante.
- Propose les événements les plus pertinents.
- Donne le titre, la date et le lieu lorsqu'ils sont disponibles.
- Réponds en français, de manière claire et concise.
""".strip(),
        ),
        (
            "human",
            f"""
Question :
{question}

Contexte :
{context}
""".strip(),
        ),
    ]

    response = llm.invoke(messages)

    sources = [
        {
            "uid": document.metadata.get("uid"),
            "title": document.metadata.get("title"),
            "begin": document.metadata.get("begin"),
            "location_name": document.metadata.get("location_name"),
            "city": document.metadata.get("city"),
        }
        for document in documents
    ]

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }


if __name__ == "__main__":
    user_question = input("Pose ta question : ")
    result = ask_chatbot(user_question)

    print("\nRéponse :\n")
    print(result["answer"])

    print("\nSources récupérées :")
    for source in result["sources"]:
        print(
            f"- {source['title']} | "
            f"{source['begin']} | "
            f"{source['city']}"
        )