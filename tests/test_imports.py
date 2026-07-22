def test_core_imports():
    """
    Vérifie que les principales dépendances du système RAG
    peuvent être importées dans l'environnement.
    """

    import faiss
    import torch

    from langchain_community.vectorstores import FAISS
    from langchain_mistralai import MistralAIEmbeddings
    from mistralai.client import Mistral
    from transformers import pipeline

    assert faiss is not None
    assert torch is not None
    assert FAISS is not None
    assert MistralAIEmbeddings is not None
    assert Mistral is not None
    assert pipeline is not None