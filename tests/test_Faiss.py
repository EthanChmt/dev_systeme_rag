import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"

# Chargement des variables d'environnement locales.
load_dotenv(dotenv_path=ENV_PATH)

EMBEDDING_MODEL = os.getenv(
    "MISTRAL_EMBEDDING_MODEL",
    "mistral-embed",
)


def test_faiss_index_exists():
    """
    Vérifie que l'index FAISS et ses fichiers
    de persistance ont bien été générés.
    """

    assert INDEX_PATH.exists(), (
        f"Le dossier d'index est absent : {INDEX_PATH}"
    )

    assert (INDEX_PATH / "index.faiss").exists(), (
        "Le fichier index.faiss est absent."
    )

    assert (INDEX_PATH / "index.pkl").exists(), (
        "Le fichier index.pkl est absent."
    )


def test_faiss_similarity_search():
    """
    Vérifie qu'une recherche sémantique peut être réalisée
    avec le même modèle Mistral que lors de l'indexation.
    """

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        pytest.skip(
            "MISTRAL_API_KEY absente : "
            "test de recherche sémantique ignoré."
        )

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vectorstore.similarity_search(
        "visite patrimoine musée concert exposition",
        k=3,
    )

    assert results, (
        "La recherche FAISS n'a retourné aucun résultat."
    )

    first_result = results[0]

    assert first_result.page_content.strip(), (
        "Le premier résultat ne contient aucun texte."
    )

    assert "title" in first_result.metadata, (
        "La métadonnée title est absente."
    )

    assert first_result.metadata["title"].strip(), (
        "La métadonnée title est vide."
    )