def test_core_imports():
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from mistralai.client import Mistral

    assert faiss is not None
    assert FAISS is not None
    assert HuggingFaceEmbeddings is not None
    assert Mistral is not None
