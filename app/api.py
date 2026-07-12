from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.chatbot import ask_chatbot


app = FastAPI(
    title="API RAG Puls-Events",
    description="API de recommandation d'événements basée sur FAISS et Mistral.",
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


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="La question ne peut pas être vide.",
            )

        result = ask_chatbot(question)

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question : {error}",
        ) from error