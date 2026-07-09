from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_PATH = Path("vectorstore/faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_faiss_index_exists():
    assert INDEX_PATH.exists()
    assert (INDEX_PATH / "index.faiss").exists()
    assert (INDEX_PATH / "index.pkl").exists()


def test_faiss_similarity_search():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vectorstore.similarity_search("concert musique théâtre exposition", k=3)

    assert len(results) > 0
    assert results[0].page_content
    assert "title" in results[0].metadata