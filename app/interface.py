import html
import os
from datetime import datetime

import gradio as gr
import httpx

from app.interface_style import (
    CUSTOM_CSS,
    PACA_THEME,
)


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 600.0


def format_event_date(
    value: str | None,
) -> str:
    """Transforme une date ISO en date lisible."""

    if not value:
        return "Date non renseignée"

    try:
        parsed_date = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )

        return parsed_date.strftime(
            "%d/%m/%Y à %H:%M"
        )

    except (TypeError, ValueError):
        return str(value)


def format_sources(
    sources: list[dict],
) -> str:
    """Présente les sources sous forme de cartes."""

    if not sources:
        return """
        <div class="empty-state">
            Aucun événement source n'a été retourné.
        </div>
        """

    cards = []

    for position, source in enumerate(
        sources,
        start=1,
    ):
        title = html.escape(
            str(
                source.get("title")
                or "Événement sans titre"
            )
        )

        city = html.escape(
            str(
                source.get("city")
                or "Ville non renseignée"
            )
        )

        location = html.escape(
            str(
                source.get("location_name")
                or "Lieu non renseigné"
            )
        )

        address = html.escape(
            str(
                source.get("address")
                or "Adresse non renseignée"
            )
        )

        begin = html.escape(
            format_event_date(
                source.get("begin")
            )
        )

        end_value = source.get("end")
        date_display = begin

        if end_value:
            end = html.escape(
                format_event_date(end_value)
            )

            if end != begin:
                date_display = (
                    f"{begin}<br>au {end}"
                )

        cards.append(
            f"""
            <article class="source-card">
                <div class="source-header">
                    <span class="source-number">
                        Source {position}
                    </span>

                    <span class="source-city">
                        {city}
                    </span>
                </div>

                <h3>{title}</h3>

                <div class="source-detail">
                    <strong>Date</strong><br>
                    {date_display}
                </div>

                <div class="source-detail">
                    <strong>Lieu</strong><br>
                    {location}
                </div>

                <div class="source-detail">
                    <strong>Adresse</strong><br>
                    {address}
                </div>
            </article>
            """
        )

    return (
        '<div class="sources-grid">'
        + "".join(cards)
        + "</div>"
    )


def call_ask_endpoint(
    question: str,
) -> tuple[str, str, str]:
    """
    Appelle réellement l'endpoint POST /ask.
    """

    question = str(
        question or ""
    ).strip()

    if not question:
        return (
            "",
            """
            <div class="empty-state">
                Les événements utilisés apparaîtront ici.
            </div>
            """,
            """
            <div class="notice notice-warning">
                Saisissez une question avant de lancer
                la recherche.
            </div>
            """,
        )

    try:
        response = httpx.post(
            f"{API_BASE_URL}/ask",
            json={
                "question": question,
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

        answer = str(
            result.get(
                "answer",
                "Aucune réponse générée.",
            )
        )

        sources = result.get(
            "sources",
            [],
        )

        source_count = len(sources)

        status = f"""
        <div class="notice notice-success">
            Appel POST /ask réussi.
            Réponse fondée sur {source_count}
            événement{"s" if source_count != 1 else ""}.
        </div>
        """

        return (
            answer,
            format_sources(sources),
            status,
        )

    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get(
                "detail",
                str(error),
            )
        except ValueError:
            detail = str(error)

        return (
            "",
            """
            <div class="empty-state">
                Les sources ne sont pas disponibles.
            </div>
            """,
            f"""
            <div class="notice notice-error">
                Erreur de l'API : {html.escape(str(detail))}
            </div>
            """,
        )

    except httpx.RequestError as error:
        return (
            "",
            """
            <div class="empty-state">
                Les sources ne sont pas disponibles.
            </div>
            """,
            f"""
            <div class="notice notice-error">
                Impossible de joindre FastAPI :
                {html.escape(str(error))}
            </div>
            """,
        )


def call_health_endpoint() -> str:
    """
    Appelle réellement l'endpoint GET /health.
    """

    try:
        response = httpx.get(
            f"{API_BASE_URL}/health",
            timeout=10.0,
        )

        response.raise_for_status()
        data = response.json()

        api_is_ok = (
            data.get("status") == "ok"
        )

        api_class = (
            "status-ok"
            if api_is_ok
            else "status-error"
        )

        api_text = (
            "API FastAPI opérationnelle"
            if api_is_ok
            else "Réponse API inattendue"
        )

        return f"""
        <div class="status-grid">
            <div class="status-item">
                <span class="status-dot {api_class}"></span>
                <span>{api_text}</span>
            </div>

            <div class="status-item">
                <span class="status-dot status-ok"></span>
                <span>Endpoint GET /health accessible</span>
            </div>

            <div class="status-item">
                <span class="status-dot status-ok"></span>
                <span>Interface Gradio active</span>
            </div>
        </div>
        """

    except Exception as error:
        return f"""
        <div class="status-grid">
            <div class="status-item">
                <span class="status-dot status-error"></span>
                <span>
                    API indisponible :
                    {html.escape(str(error))}
                </span>
            </div>
        </div>
        """


def call_rebuild_endpoint() -> tuple[str, str, str]:
    """
    Appelle réellement l'endpoint POST /rebuild.
    """

    try:
        response = httpx.post(
            f"{API_BASE_URL}/rebuild",
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

        fetch_output = result.get(
            "fetch_output",
            "",
        )

        build_output = result.get(
            "build_output",
            "",
        )

        logs = (
            "APPEL POST /rebuild\n"
            "====================\n\n"
            "COLLECTE OPENAGENDA\n"
            "-------------------\n"
            f"{fetch_output}\n\n"
            "CONSTRUCTION DE L'INDEX FAISS\n"
            "-----------------------------\n"
            f"{build_output}"
        )

        status = """
        <div class="notice notice-success">
            L'endpoint POST /rebuild a terminé
            correctement la reconstruction.
        </div>
        """

        return (
            logs,
            status,
            call_health_endpoint(),
        )

    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get(
                "detail",
                str(error),
            )
        except ValueError:
            detail = str(error)

        return (
            f"Échec de POST /rebuild.\n\n{detail}",
            f"""
            <div class="notice notice-error">
                Reconstruction échouée :
                {html.escape(str(detail))}
            </div>
            """,
            call_health_endpoint(),
        )

    except httpx.RequestError as error:
        return (
            f"FastAPI est inaccessible.\n\n{error}",
            f"""
            <div class="notice notice-error">
                Impossible de joindre l'endpoint /rebuild.
            </div>
            """,
            call_health_endpoint(),
        )


def create_interface() -> gr.Blocks:
    """Construit l'interface Gradio PACA."""

    with gr.Blocks(
        title="Puls-Events PACA",
        theme=PACA_THEME,
        css=CUSTOM_CSS,
        fill_width=True,
    ) as interface:

        gr.HTML(
            """
            <section class="hero-paca">
                <div class="hero-content">
                    <span class="hero-kicker">
                        Culture · Patrimoine · Région PACA
                    </span>

                    <h1>
                        Explorez le patrimoine de
                        Provence-Alpes-Côte d’Azur
                    </h1>

                    <p>
                        Posez votre question en langage
                        naturel et découvrez les événements
                        sélectionnés par notre système RAG.
                    </p>

                    <div class="hero-tags">
                        <span>OpenAgenda</span>
                        <span>Mistral Embeddings</span>
                        <span>FAISS</span>
                        <span>Hugging Face</span>
                        <span>JEP 2026</span>
                    </div>
                </div>
            </section>
            """
        )

        with gr.Group(
            elem_classes=["paca-card"],
        ):
            gr.HTML(
                """
                <div class="section-title">
                    Trouver un événement
                </div>

                <div class="section-description">
                    Précisez une ville, un département,
                    un type de monument ou une activité.
                </div>
                """
            )

            question_input = gr.Textbox(
                label="Votre question",
                placeholder=(
                    "Quelles visites historiques sont "
                    "proposées à Marseille ?"
                ),
                lines=3,
                autofocus=True,
            )

            with gr.Row():
                search_button = gr.Button(
                    "Rechercher",
                    variant="primary",
                )

                clear_button = gr.ClearButton(
                    value="Effacer",
                    components=[
                        question_input,
                    ],
                )

            gr.Examples(
                examples=[
                    [
                        "Quelles visites historiques sont "
                        "proposées à Marseille ?"
                    ],
                    [
                        "Quels monuments religieux peut-on "
                        "visiter dans le Var ?"
                    ],
                    [
                        "Trouve-moi une activité familiale "
                        "dans les Alpes-Maritimes."
                    ],
                    [
                        "Quelles visites guidées sont "
                        "disponibles à Aix-en-Provence ?"
                    ],
                ],
                inputs=question_input,
                label="Exemples",
            )

        with gr.Group(
            elem_classes=["paca-card"],
        ):
            request_status = gr.HTML(
                value="""
                <div class="notice notice-info">
                    La recherche appellera l'endpoint
                    POST /ask.
                </div>
                """
            )

            answer_output = gr.Textbox(
                label="Réponse",
                lines=8,
                interactive=False,
                elem_classes=["answer-output"],
            )

        with gr.Group(
            elem_classes=["paca-card"],
        ):
            gr.HTML(
                """
                <div class="section-title">
                    Événements sources
                </div>

                <div class="section-description">
                    Événements retournés par l'API
                    et utilisés pour construire la réponse.
                </div>
                """
            )

            sources_output = gr.HTML(
                value="""
                <div class="empty-state">
                    Les sources apparaîtront ici.
                </div>
                """
            )

        with gr.Accordion(
            "État du système et administration",
            open=False,
        ):
            system_status = gr.HTML(
                value="""
                <div class="notice notice-info">
                    Cliquez sur « Vérifier l’API » pour
                    appeler GET /health.
                </div>
                """
            )

            health_button = gr.Button(
                "Vérifier l’API",
            )

            gr.HTML(
                """
                <div class="api-links">
                    <a href="/docs" target="_blank">
                        Swagger
                    </a>

                    <a href="/health" target="_blank">
                        Réponse brute de /health
                    </a>
                </div>
                """
            )

            rebuild_button = gr.Button(
                "Reconstruire la base vectorielle",
            )

            rebuild_status = gr.HTML()

            rebuild_logs = gr.Textbox(
                label="Journal de POST /rebuild",
                lines=12,
                interactive=False,
            )

        gr.HTML(
            """
            <div class="footer-paca">
                Puls-Events · POC RAG culturel local<br>
                Provence-Alpes-Côte d’Azur
            </div>
            """
        )

        search_button.click(
            fn=call_ask_endpoint,
            inputs=question_input,
            outputs=[
                answer_output,
                sources_output,
                request_status,
            ],
        )

        question_input.submit(
            fn=call_ask_endpoint,
            inputs=question_input,
            outputs=[
                answer_output,
                sources_output,
                request_status,
            ],
        )

        health_button.click(
            fn=call_health_endpoint,
            inputs=[],
            outputs=system_status,
        )

        rebuild_button.click(
            fn=call_rebuild_endpoint,
            inputs=[],
            outputs=[
                rebuild_logs,
                rebuild_status,
                system_status,
            ],
        )

    return interface