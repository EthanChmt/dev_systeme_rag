import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

DATA_PATH = PROJECT_ROOT / "data" / "events_openagenda.csv"
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"

load_dotenv(dotenv_path=ENV_PATH)

EMBEDDING_MODEL = os.getenv(
    "MISTRAL_EMBEDDING_MODEL",
    "mistral-embed",
)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    Transforme les événements du fichier CSV en documents
    LangChain avec leurs métadonnées.
    """

    documents = []

    for _, row in df.iterrows():
        text = str(
            row.get("text_for_embedding", "")
        ).strip()

        if not text:
            continue

        metadata = {
            "uid": str(row.get("uid", "")),
            "title": str(row.get("title", "")),
            "begin": str(row.get("begin", "")),
            "end": str(row.get("end", "")),
            "location_name": str(
                row.get("location_name", "")
            ),
            "city": str(row.get("city", "")),
            "address": str(row.get("address", "")),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


def build_faiss_index() -> None:
    """
    Génère les embeddings Mistral et construit
    l'index vectoriel FAISS.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH).fillna("")
    documents = build_documents(df)

    if not documents:
        raise ValueError(
            "Aucun document exploitable pour l'indexation."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    print(
        f"Génération des embeddings avec "
        f"{EMBEDDING_MODEL}..."
    )

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
    )

    INDEX_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(INDEX_PATH)
    )

    print(f"{len(documents)} événements chargés.")
    print(f"{len(chunks)} chunks indexés.")
    print(f"Modèle d'embedding : {EMBEDDING_MODEL}")
    print(f"Index FAISS sauvegardé dans : {INDEX_PATH}")


if __name__ == "__main__":
    build_faiss_index()