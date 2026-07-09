from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = Path("data/events_openagenda.csv")
INDEX_PATH = Path("vectorstore/faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_documents(df: pd.DataFrame):
    documents = []

    for _, row in df.iterrows():
        text = str(row.get("text_for_embedding", "")).strip()

        if not text:
            continue

        metadata = {
            "uid": str(row.get("uid", "")),
            "title": str(row.get("title", "")),
            "begin": str(row.get("begin", "")),
            "end": str(row.get("end", "")),
            "location_name": str(row.get("location_name", "")),
            "city": str(row.get("city", "")),
            "address": str(row.get("address", "")),
        }

        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def build_faiss_index():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).fillna("")
    documents = build_documents(df)

    if not documents:
        raise ValueError("Aucun document exploitable pour l'indexation.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))

    print(f"{len(documents)} événements chargés.")
    print(f"{len(chunks)} chunks indexés.")
    print(f"Index FAISS sauvegardé dans : {INDEX_PATH}")


if __name__ == "__main__":
    build_faiss_index()