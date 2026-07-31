import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from transformers import pipeline

import torch
# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index"

load_dotenv(dotenv_path=ENV_PATH)

EMBEDDING_MODEL = os.getenv(
    "MISTRAL_EMBEDDING_MODEL",
    "mistral-embed",
)

GENERATION_MODEL = os.getenv(
    "HUGGINGFACE_GENERATION_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)

# Nombre maximal d'événements proposés.
MAX_EVENTS_IN_CONTEXT = 3

# Le modèle Hugging Face reste chargé en mémoire
# après sa première utilisation.
TEXT_GENERATOR = None


# ============================================================
# CHARGEMENT DES COMPOSANTS
# ============================================================

def load_vectorstore():
    """
    Charge l'index FAISS avec le modèle d'embedding Mistral
    utilisé lors de sa construction.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_PATH}. "
            "Lance d'abord scripts/build_faiss_index.py."
        )

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    print("1. Initialisation du modèle d'embedding Mistral...")

    embeddings = MistralAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
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


def load_text_generator():
    """
    Charge le modèle Hugging Face sur le GPU NVIDIA
    lorsqu'il est disponible, avec repli automatique sur le CPU.
    """

    global TEXT_GENERATOR

    if TEXT_GENERATOR is not None:
        return TEXT_GENERATOR

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)

        print(
            "GPU CUDA détecté : "
            f"{device_name}"
        )

        device = 0
        model_dtype = torch.float16

    else:
        print(
            "Aucun GPU CUDA détecté. "
            "Utilisation du CPU."
        )

        device = -1
        model_dtype = torch.float32

    print(
        "Chargement du modèle de génération : "
        f"{GENERATION_MODEL}"
    )

    TEXT_GENERATOR = pipeline(
        task="text-generation",
        model=GENERATION_MODEL,
        tokenizer=GENERATION_MODEL,
        device=device,
        torch_dtype=model_dtype,
    )

    print(
        "Modèle Hugging Face chargé sur "
        f"{'GPU' if cuda_available else 'CPU'}."
    )

    return TEXT_GENERATOR


# ============================================================
# TRAITEMENT DES DATES ET DES ÉVÉNEMENTS
# ============================================================

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
            date_value = date_value.replace(
                tzinfo=timezone.utc
            )

        return date_value

    except (TypeError, ValueError):
        return None


def filter_future_documents(documents):
    """
    Conserve uniquement les événements futurs et retire
    les doublons.

    L'ordre retourné par FAISS est conservé afin de préserver
    le classement par pertinence sémantique.
    """

    now = datetime.now(timezone.utc)
    future_documents = []
    seen_events = set()

    for document in documents:
        metadata = document.metadata

        begin_date = parse_event_date(
            metadata.get("begin")
        )

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

    # Aucun tri par date ici :
    # l'ordre de pertinence FAISS doit être conservé.
    return future_documents


def format_context(documents):
    """
    Transforme les événements récupérés en contexte lisible
    pour le modèle de génération.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
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


def build_availability_instruction(
    event_count: int,
) -> str:
    """
    Construit la consigne liée au nombre d'événements
    disponibles.
    """

    if event_count == 1:
        return (
            "Un seul événement est disponible. "
            "Présente uniquement cet événement et précise "
            "naturellement qu'il s'agit du seul événement "
            "trouvé."
        )

    if event_count == 2:
        return (
            "Deux événements sont disponibles. "
            "Présente les deux événements et précise "
            "naturellement qu'il s'agit des deux seuls "
            "événements trouvés."
        )

    return (
        "Trois événements sont disponibles. "
        "Présente les trois événements dans une réponse "
        "fluide et structurée."
    )


def add_limited_results_notice(
    answer: str,
    event_count: int,
) -> str:
    """
    Ajoute automatiquement une précision lorsque moins
    de trois événements sont disponibles.

    Cette étape ne dépend pas du respect du prompt
    par le modèle local.
    """

    answer = answer.strip()

    if event_count == 1:
        notice = (
            "Je n’ai trouvé qu’un seul événement "
            "correspondant à votre recherche."
        )

        if notice.lower() not in answer.lower():
            return f"{notice}\n\n{answer}"

    if event_count == 2:
        notice = (
            "Je n’ai trouvé que deux événements "
            "correspondant à votre recherche."
        )

        if notice.lower() not in answer.lower():
            return f"{notice}\n\n{answer}"

    return answer


# ============================================================
# EXTRACTION DE LA RÉPONSE HUGGING FACE
# ============================================================

def extract_generated_answer(generated_output):
    """
    Extrait le texte final retourné par la pipeline
    Hugging Face.
    """

    if not generated_output:
        return ""

    generated_text = generated_output[0].get(
        "generated_text",
        "",
    )

    if isinstance(generated_text, list):
        for message in reversed(generated_text):
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
            ):
                return str(
                    message.get("content", "")
                ).strip()

        if generated_text:
            last_message = generated_text[-1]

            if isinstance(last_message, dict):
                return str(
                    last_message.get("content", "")
                ).strip()

    return str(generated_text).strip()


# ============================================================
# PIPELINE RAG
# ============================================================

def ask_chatbot(question: str) -> dict:
    """
    Recherche les événements pertinents dans FAISS,
    conserve les événements futurs puis génère une réponse
    avec le modèle Hugging Face local.
    """

    question = str(question or "").strip()

    if not question:
        raise ValueError(
            "La question ne peut pas être vide."
        )

    print("5. Chargement de la base vectorielle...")
    vectorstore = load_vectorstore()

    print("6. Recherche sémantique dans FAISS...")

    candidate_documents = vectorstore.similarity_search(
        question,
        k=100,
    )

    print(
        f"7. {len(candidate_documents)} chunks candidats "
        "récupérés."
    )

    documents = filter_future_documents(
        candidate_documents
    )

    # On conserve les trois premiers événements uniques
    # dans l'ordre de pertinence FAISS.
    documents = documents[
        :MAX_EVENTS_IN_CONTEXT
    ]

    event_count = len(documents)

    print(
        f"8. {event_count} événements futurs et uniques "
        "conservés."
    )

    if not documents:
        return {
            "question": question,
            "answer": (
                "Je ne dispose pas d’événement futur "
                "suffisamment pertinent pour répondre "
                "à cette question."
            ),
            "sources": [],
        }

    availability_instruction = (
        build_availability_instruction(
            event_count
        )
    )

    print("9. Construction du contexte...")

    context = format_context(
        documents
    )

    print("10. Contexte construit.")

    current_date = datetime.now().strftime(
        "%d/%m/%Y"
    )

    system_prompt = f"""
Tu es un assistant spécialisé dans les recommandations culturelles.

La date actuelle est le {current_date}.

Tu dois répondre précisément à la question de l'utilisateur
en utilisant uniquement les événements présents dans le contexte.

Les événements sont classés par ordre de pertinence sémantique.

Nombre d'événements disponibles : {event_count}.

Consigne concernant le nombre de recommandations :
{availability_instruction}

Règles obligatoires :
- Commence par répondre directement à la question posée.
- Présente uniquement les événements réellement en rapport avec la demande.
- Utilise les événements dans l'ordre où ils sont fournis.
- Lorsque trois événements sont disponibles, présente les trois.
- Lorsque deux événements sont disponibles, présente les deux.
- Lorsqu'un seul événement est disponible, présente uniquement celui-ci.
- N'invente aucun événement.
- N'invente aucune date, aucun lieu, aucun tarif ou aucune activité.
- Ne propose jamais un événement dont la date est passée.
- Utilise uniquement les informations présentes dans le contexte.
- Pour chaque événement présenté, indique naturellement son titre, sa date, sa ville et son lieu.
- Si une information n'est pas présente dans le contexte, ne la complète pas.
- Réponds en français.
- Rédige un ou deux paragraphes fluides, naturels et bien structurés.
- Utilise des phrases complètes.
- Ne fais aucune liste.
- N'utilise aucune puce.
- N'utilise aucune numérotation.
- N'utilise aucun titre.
- N'utilise pas de Markdown.
- Ne répète pas les mêmes informations.
""".strip()

    human_prompt = f"""
Question de l'utilisateur :

{question}

Événements récupérés par ordre de pertinence dans FAISS :

{context}

Réponds précisément à la question dans un ou deux paragraphes
fluides. Ne transforme pas la réponse en liste.
""".strip()

    generator = load_text_generator()

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": human_prompt,
        },
    ]

    print("11. Génération locale de la réponse...")

    generated_output = generator(
        messages,
        max_new_tokens=1000,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    answer = extract_generated_answer(
        generated_output
    )

    if not answer:
        raise RuntimeError(
            "Le modèle Hugging Face n'a généré "
            "aucune réponse exploitable."
        )

    answer = add_limited_results_notice(
        answer=answer,
        event_count=event_count,
    )

    print("12. Réponse Hugging Face générée.")

    sources = []

    for document in documents:
        metadata = document.metadata

        sources.append(
            {
                "uid": metadata.get("uid"),
                "title": metadata.get("title"),
                "begin": metadata.get("begin"),
                "end": metadata.get("end"),
                "location_name": metadata.get(
                    "location_name"
                ),
                "city": metadata.get("city"),
                "address": metadata.get(
                    "address"
                ),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# UTILISATION DANS LE TERMINAL
# ============================================================

def main():
    """Lance le chatbot directement dans le terminal."""

    try:
        user_question = input(
            "Pose ta question : "
        ).strip()

        print(
            "\nDémarrage du pipeline RAG...\n"
        )

        result = ask_chatbot(
            user_question
        )

        print("\n" + "=" * 60)
        print("RÉPONSE")
        print("=" * 60)
        print(result["answer"])

        print("\n" + "=" * 60)
        print("SOURCES RÉCUPÉRÉES PAR FAISS")
        print("=" * 60)

        if not result["sources"]:
            print(
                "Aucune source future récupérée."
            )
            return

        for index, source in enumerate(
            result["sources"],
            start=1,
        ):
            print(f"\nSource {index}")
            print(
                f"Titre : {source['title']}"
            )
            print(
                f"Début : {source['begin']}"
            )
            print(
                f"Fin : {source['end']}"
            )
            print(
                f"Lieu : {source['location_name']}"
            )
            print(
                f"Ville : {source['city']}"
            )
            print(
                f"Adresse : {source['address']}"
            )

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERREUR")
        print("=" * 60)
        print(
            f"Type : {type(error).__name__}"
        )
        print(
            f"Message : {error}"
        )


if __name__ == "__main__":
    main()