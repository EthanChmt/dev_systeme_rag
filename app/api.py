import subprocess
import sys
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.chatbot import ask_chatbot


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_openagenda_events.py"
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_faiss_index.py"


app = FastAPI(
    title="API RAG Puls-Events",
    description=(
        "API de recommandation d'événements basée sur "
        "OpenAgenda, FAISS et Mistral."
    ),
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class SourceResponse(BaseModel):
    uid: str | None = None
    title: str | None = None
    begin: str | None = None
    end: str | None = None
    location_name: str | None = None
    city: str | None = None
    address: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]


class RebuildResponse(BaseModel):
    status: str
    message: str
    fetch_output: str
    build_output: str


def run_script(script_path: Path) -> str:
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script introuvable : {script_path}"
        )

    process_env = os.environ.copy()
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONUTF8"] = "1"

    process = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
        env=process_env,
    )

    output = process.stdout.strip()
    error_output = process.stderr.strip()

    if process.returncode != 0:
        raise RuntimeError(
            f"Échec du script {script_path.name}.\n"
            f"Sortie :\n{output}\n"
            f"Erreur :\n{error_output}"
        )

    if error_output:
        output = (
            f"{output}\n\n"
            f"Avertissements :\n{error_output}"
        ).strip()

    return output


@app.get("/")
def root():
    return {
        "message": "API RAG Puls-Events opérationnelle.",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask(request: QuestionRequest):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="La question ne peut pas être vide.",
            )

        return ask_chatbot(question)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors du traitement de la question : "
                f"{error}"
            ),
        ) from error


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
)
def rebuild():
    """
    Récupère les événements OpenAgenda puis
    reconstruit entièrement l'index FAISS.
    """

    try:
        fetch_output = run_script(FETCH_SCRIPT)
        build_output = run_script(BUILD_SCRIPT)

        return {
            "status": "success",
            "message": (
                "Les données OpenAgenda et l'index FAISS "
                "ont été reconstruits avec succès."
            ),
            "fetch_output": fetch_output,
            "build_output": build_output,
        }

    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail=(
                "La reconstruction a dépassé le délai "
                "maximum autorisé."
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la reconstruction : {error}",
        ) from error