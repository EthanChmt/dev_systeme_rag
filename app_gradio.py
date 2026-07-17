import gradio as gr

from app.chatbot import ask_chatbot


def respond(question: str) -> str:
    """
    Transmet la question au système RAG et retourne
    uniquement la réponse générée.
    """

    question = question.strip()

    if not question:
        return "Veuillez saisir une question."

    try:
        result = ask_chatbot(question)
        return result["answer"]

    except Exception as error:
        return f"Une erreur est survenue : {error}"


demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(
        label="Votre question",
        placeholder=(
            "Exemple : Quels sont les prochains événements "
            "musicaux dans la région ?"
        ),
        lines=2,
    ),
    outputs=gr.Textbox(
        label="Réponse",
        lines=10,
    ),
    title="Puls-Events",
    description=(
        "Assistant de recommandation d’événements basé sur "
        "OpenAgenda, FAISS et Mistral."
    ),
    examples=[
        ["Quels sont les prochains événements musicaux dans la région ?"],
        ["Quelles visites sont prévues prochainement ?"],
        ["Quels événements ont lieu à Marseille ?"],
    ],
)


if __name__ == "__main__":
    demo.launch()