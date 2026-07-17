from fastapi.testclient import TestClient

from app.api import app


# Client de test permettant d'appeler l'API sans démarrer Uvicorn.
client = TestClient(app)


def test_health():
    """Vérifie que l'endpoint de santé répond correctement."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    """Vérifie le contenu de la route racine de l'API."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["documentation"] == "/docs"


def test_ask_with_empty_question():
    """Vérifie qu'une question vide génère une erreur HTTP 400."""

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "La question ne peut pas être vide."
    )


def test_ask_with_valid_question(monkeypatch):
    """
    Vérifie qu'une question valide retourne une réponse conforme
    sans effectuer d'appel réel au modèle Mistral.
    """

    # Réponse simulée du chatbot.
    fake_result = {
        "question": "Quels événements musicaux sont disponibles ?",
        "answer": (
            "Un concert est prévu prochainement à Marseille."
        ),
        "sources": [
            {
                "uid": "123",
                "title": "Concert test",
                "begin": "2026-09-18T20:00:00+00:00",
                "end": "2026-09-18T22:00:00+00:00",
                "location_name": "Salle test",
                "city": "Marseille",
                "address": "1 rue de test",
            }
        ],
    }

    # Fonction simulée utilisée à la place du chatbot réel.
    def fake_ask_chatbot(question):
        return fake_result

    # Remplacement temporaire de la fonction appelée par l'API.
    monkeypatch.setattr(
        "app.api.ask_chatbot",
        fake_ask_chatbot,
    )

    response = client.post(
        "/ask",
        json={
            "question": (
                "Quels événements musicaux sont disponibles ?"
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == fake_result


def test_rebuild(monkeypatch):
    """
    Vérifie la réponse de l'endpoint /rebuild sans exécuter
    réellement les scripts de collecte et d'indexation.
    """

    # Sorties simulées des scripts exécutés par l'endpoint.
    outputs = {
        "fetch_openagenda_events.py": (
            "300 événements sauvegardés."
        ),
        "build_faiss_index.py": (
            "Index FAISS sauvegardé."
        ),
    }

    # Fonction simulée remplaçant l'exécution des scripts.
    def fake_run_script(script_path):
        return outputs[script_path.name]

    # Remplacement temporaire de la fonction run_script.
    monkeypatch.setattr(
        "app.api.run_script",
        fake_run_script,
    )

    response = client.post("/rebuild")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert "reconstruits avec succès" in data["message"]
    assert data["fetch_output"] == (
        "300 événements sauvegardés."
    )
    assert data["build_output"] == (
        "Index FAISS sauvegardé."
    )