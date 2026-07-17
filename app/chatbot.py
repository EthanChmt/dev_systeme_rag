import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# Chemins du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"

# Chargement des variables d'environnement
load_dotenv(dotenv_path=ENV_PATH)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "ministral-3b-2512")


def load_vectorstore():
    """Charge le modèle d'embedding et l'index FAISS local."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_PATH}. "
            "Lance d'abord scripts/build_faiss_index.py."
        )

    print("1. Initialisation du modèle d'embedding...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("2. Modèle d'embedding initialisé.")
    print("3. Chargement de l'index FAISS...")

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("4. Index FAISS chargé.")

    return vectorstore


def parse_event_date(value):
    """
    Convertit une date provenant des métadonnées FAISS
    en objet datetime avec fuseau horaire.
    """

    if not value:
        return None

    try:
        date_value = datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )

        if date_value.tzinfo is None:
            date_value = date_value.replace(tzinfo=timezone.utc)

        return date_value

    except (TypeError, ValueError):
        return None


def filter_future_documents(documents):
    """
    Conserve uniquement les événements futurs et retire
    les doublons correspondant au même événement.
    """

    now = datetime.now(timezone.utc)
    future_documents = []
    seen_events = set()

    for document in documents:
        metadata = document.metadata
        begin_date = parse_event_date(metadata.get("begin"))

        if begin_date is None:
            continue

        if begin_date.astimezone(timezone.utc) < now:
            continue

        event_identifier = (
            metadata.get("uid"),
            metadata.get("title"),
            metadata.get("begin"),
        )

        if event_identifier in seen_events:
            continue

        seen_events.add(event_identifier)
        future_documents.append(document)

    future_documents.sort(
        key=lambda document: parse_event_date(
            document.metadata.get("begin")
        )
    )

    return future_documents


def format_context(documents):
    """Transforme les événements récupérés en contexte pour Mistral."""

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

Contenu :
{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


def ask_chatbot(question: str) -> dict:
    """
    Recherche les événements pertinents dans FAISS,
    filtre les événements passés puis génère une réponse avec Mistral.
    """

    question = question.strip()

    if not question:
        raise ValueError("La question ne peut pas être vide.")

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY est absente du fichier .env.")

    print("5. Chargement de la base vectorielle...")
    vectorstore = load_vectorstore()

    print("6. Recherche sémantique dans FAISS...")

    # On récupère davantage de résultats avant le filtrage temporel.
    candidate_documents = vectorstore.similarity_search(
        question,
        k=100,
    )

    print(
        f"7. {len(candidate_documents)} chunks candidats "
        "récupérés."
    )

    documents = filter_future_documents(candidate_documents)
    documents = documents[:6]

    print(
        f"8. {len(documents)} événements futurs et uniques "
        "conservés."
    )

    if not documents:
        return {
            "question": question,
            "answer": (
                "Je ne dispose pas d'événements futurs suffisamment "
                "pertinents pour répondre à cette question."
            ),
            "sources": [],
        }

    print("9. Construction du contexte...")
    context = format_context(documents)
    print("10. Contexte construit.")

    print(
        f"11. Initialisation du modèle Mistral : "
        f"{MISTRAL_MODEL}"
    )

    llm = ChatMistralAI(
        model=MISTRAL_MODEL,
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )

    print("12. Client Mistral initialisé.")

    current_date = datetime.now().strftime("%d/%m/%Y")

    system_prompt = f"""
Tu es un assistant spécialisé dans les recommandations culturelles.

La date actuelle est le {current_date}.

Tu dois répondre uniquement à partir des événements présents dans le contexte.

Règles obligatoires :
- N'invente aucun événement.
- N'invente aucune date, aucun lieu, aucun tarif ou aucune activité.
- Ne propose jamais un événement dont la date est déjà passée.
- Si le contexte ne permet pas de répondre, indique que les informations sont insuffisantes.
- Privilégie les événements qui correspondent réellement à la question.
- N'ajoute pas d'information qui ne figure pas dans le contexte.
- Réponds en français.
- Réponds avec des phrases complètes et naturelles.
- Ne présente pas la réponse sous forme de liste.
- N'utilise ni Markdown, ni astérisques, ni titres, ni *, ni /.
- Regroupe les événements dans un ou deux paragraphes fluides.
- Pour chaque événement cité, indique naturellement le titre, la date, la ville et le lieu.
- Ne répète pas les mêmes informations.
""".strip()

    human_prompt = f"""
Question de l'utilisateur :

{question}

Événements futurs récupérés dans la base FAISS :

{context}
""".strip()

    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]

    print("13. Envoi de la requête à Mistral...")

    response = llm.invoke(messages)

    print("14. Réponse reçue de Mistral.")

    sources = []

    for document in documents:
        sources.append(
            {
                "uid": document.metadata.get("uid"),
                "title": document.metadata.get("title"),
                "begin": document.metadata.get("begin"),
                "end": document.metadata.get("end"),
                "location_name": document.metadata.get(
                    "location_name"
                ),
                "city": document.metadata.get("city"),
                "address": document.metadata.get("address"),
            }
        )

    return {
        "question": question,
        "answer": response.content,
        "sources": sources,
    }


def main():
    """Lance le chatbot dans le terminal."""

    try:
        user_question = input("Pose ta question : ").strip()

        print("\nDémarrage du pipeline RAG...\n")

        result = ask_chatbot(user_question)

        print("\n" + "=" * 60)
        print("RÉPONSE")
        print("=" * 60)
        print(result["answer"])

        print("\n" + "=" * 60)
        print("SOURCES RÉCUPÉRÉES PAR FAISS")
        print("=" * 60)

        if not result["sources"]:
            print("Aucune source future récupérée.")
            return

        for index, source in enumerate(
            result["sources"],
            start=1,
        ):
            print(f"\nSource {index}")
            print(f"Titre : {source['title']}")
            print(f"Début : {source['begin']}")
            print(f"Fin : {source['end']}")
            print(f"Lieu : {source['location_name']}")
            print(f"Ville : {source['city']}")
            print(f"Adresse : {source['address']}")

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERREUR")
        print("=" * 60)
        print(f"Type : {type(error).__name__}")
        print(f"Message : {error}")


if __name__ == "__main__":
    main()