import os
import re
import html
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENAGENDA_API_KEY")
AGENDA_UID = os.getenv("OPENAGENDA_AGENDA_UID")
CITY = os.getenv("OPENAGENDA_CITY", "")
SEARCH = os.getenv("OPENAGENDA_SEARCH", "")

MAX_PAST_EVENTS = int(
    os.getenv(
        "OPENAGENDA_MAX_PAST_EVENTS",
        "100",
    )
)

MAX_FUTURE_EVENTS = int(
    os.getenv(
        "OPENAGENDA_MAX_FUTURE_EVENTS",
        "300",
    )
)

OUTPUT_PATH = Path(
    "data/events_openagenda.csv"
)


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, dict):
        value = (
            value.get("fr")
            or next(
                iter(value.values()),
                "",
            )
        )

    if isinstance(value, list):
        value = ", ".join(
            str(item)
            for item in value
        )

    value = str(value)
    value = html.unescape(value)
    value = re.sub(
        r"<[^>]+>",
        " ",
        value,
    )
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def timing_in_range(event, start_date, end_date):
    """
    Renvoie le premier créneau de l'événement qui tombe
    réellement dans [start_date, end_date].

    event.get("timings") liste TOUS les créneaux de
    l'événement, pas seulement ceux qui matchent le filtre
    timings[gte]/timings[lte] envoyé à l'API (ce filtre ne
    sert qu'à sélectionner quels événements remonter, pas à
    trier ce tableau). Prendre timings[0] à l'aveugle peut
    donc renvoyer une date hors de la fenêtre demandée.
    """

    timings = event.get("timings") or []

    candidates = []

    for t in timings:
        begin_raw = t.get("begin")

        if not begin_raw:
            continue

        try:
            begin_dt = datetime.fromisoformat(
                begin_raw.replace("Z", "+00:00")
            )
        except ValueError:
            continue

        if start_date <= begin_dt <= end_date:
            candidates.append((begin_dt, t))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    _, chosen = candidates[0]

    return (
        chosen.get("begin", ""),
        chosen.get("end", ""),
    )


def normalize_event(event, start_date, end_date):
    location = event.get("location") or {}

    timing = timing_in_range(event, start_date, end_date)

    if timing is None:
        # Aucun créneau de l'événement ne tombe réellement
        # dans la fenêtre demandée : on l'ignore plutôt que
        # de renvoyer une date hors périmètre.
        return None

    begin, end = timing

    title = clean_text(
        event.get("title")
    )

    description = clean_text(
        event.get("description")
    )

    long_description = clean_text(
        event.get("longDescription")
    )

    conditions = clean_text(
        event.get("conditions")
    )

    keywords = clean_text(
        event.get("keywords")
    )

    location_name = clean_text(
        location.get("name")
    )

    address = clean_text(
        location.get("address")
    )

    city = clean_text(
        location.get("city")
        or location.get("adminLevel4")
    )

    department = clean_text(
        location.get("department")
        or location.get("adminLevel2")
    )

    region = clean_text(
        location.get("region")
        or location.get("adminLevel1")
    )

    country = clean_text(
        location.get("country")
    )

    text_for_embedding = "\n".join(
        [
            f"Titre : {title}",
            (
                "Description courte : "
                f"{description}"
            ),
            (
                "Description longue : "
                f"{long_description}"
            ),
            f"Conditions : {conditions}",
            f"Mots-clés : {keywords}",
            (
                "Lieu : "
                f"{location_name}, "
                f"{address}, "
                f"{city}, "
                f"{department}, "
                f"{region}, "
                f"{country}"
            ),
            f"Dates : {begin} - {end}",
        ]
    )

    return {
        "uid": event.get("uid"),
        "slug": event.get("slug"),
        "title": title,
        "description": description,
        "long_description": (
            long_description
        ),
        "conditions": conditions,
        "keywords": keywords,
        "begin": begin,
        "end": end,
        "location_name": (
            location_name
        ),
        "address": address,
        "city": city,
        "department": department,
        "region": region,
        "country": country,
        "latitude": (
            location.get("latitude")
        ),
        "longitude": (
            location.get("longitude")
        ),
        "text_for_embedding": clean_text(
            text_for_embedding
        ),
    }


def fetch_event_period(
    start_date,
    end_date,
    max_events,
    sort_order,
):
    """
    Récupère une période précise dans OpenAgenda et renvoie
    directement les lignes normalisées et VALIDES (créneau
    réellement dans [start_date, end_date]).

    On continue à paginer tant qu'on n'a pas max_events
    lignes valides (et pas seulement max_events événements
    bruts), car certains événements récupérés sont ensuite
    écartés faute de créneau dans la fenêtre demandée.

    sort_order doit être :
    - timings.asc pour les futurs ;
    - timings.desc pour les passés.
    """

    url = (
        "https://api.openagenda.com/v2/"
        f"agendas/{AGENDA_UID}/events"
    )

    headers = {
        "key": API_KEY,
    }

    valid_rows = []
    skipped = 0
    after = None

    while len(valid_rows) < max_events:
        params = [
            (
                "size",
                300,
            ),
            (
                "detailed",
                "1",
            ),
            (
                "monolingual",
                "fr",
            ),
            (
                "sort",
                sort_order,
            ),
            (
                "timings[gte]",
                start_date.isoformat().replace(
                    "+00:00",
                    "Z",
                ),
            ),
            (
                "timings[lte]",
                end_date.isoformat().replace(
                    "+00:00",
                    "Z",
                ),
            ),
        ]

        if CITY:
            params.append(
                (
                    "adminLevel4[]",
                    CITY,
                )
            )

        if SEARCH:
            params.append(
                (
                    "search",
                    SEARCH,
                )
            )

            params.append(
                (
                    "threshold",
                    "auto",
                )
            )

        if after:
            for value in after:
                params.append(
                    (
                        "after[]",
                        value,
                    )
                )

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()
        batch = payload.get(
            "events",
            [],
        )

        if not batch:
            break

        for event in batch:
            if len(valid_rows) >= max_events:
                break

            row = normalize_event(
                event, start_date, end_date
            )

            if row is None:
                skipped += 1
                continue

            valid_rows.append(row)

        after = payload.get("after")

        if not after:
            break

    if skipped:
        print(
            f"{skipped} événement(s) ignoré(s) sur cette "
            "période : aucun créneau dans la fenêtre "
            "demandée (pages supplémentaires récupérées "
            "pour compenser)."
        )

    return valid_rows


def clean_dataframe(rows):
    """
    Nettoie et normalise le DataFrame final.
    """

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Aucun événement récupéré. "
            "Vérifie la clé API, l'Agenda UID, "
            "la période et les filtres."
        )

    initial_count = len(df)

    df = df.drop_duplicates(
        subset=["uid"]
    )

    df = df[
        df["title"].str.strip() != ""
    ]

    df["begin"] = pd.to_datetime(
        df["begin"],
        errors="coerce",
        utc=True,
    )

    df["end"] = pd.to_datetime(
        df["end"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["begin"]
    )

    df = df.sort_values(
        by="begin"
    )

    return df, initial_count


def print_quality_report(
    df,
    initial_count,
    past_count,
    future_count,
):
    """
    Affiche le rapport qualité de la collecte.
    """

    final_count = len(df)

    missing_description = (
        df["description"] == ""
    ).sum()

    missing_long_description = (
        df["long_description"] == ""
    ).sum()

    missing_location = (
        df["location_name"] == ""
    ).sum()

    missing_city = (
        df["city"] == ""
    ).sum()

    min_date = df["begin"].min()
    max_date = df["begin"].max()

    def percentage(value, total):
        if not total:
            return 0

        return round(
            (value / total) * 100,
            2,
        )

    print("\n" + "=" * 50)
    print("RAPPORT QUALITÉ OPENAGENDA")
    print("=" * 50)

    print(
        "Événements passés récupérés : "
        f"{past_count}"
    )

    print(
        "Événements futurs récupérés : "
        f"{future_count}"
    )

    print(
        "Événements récupérés au total : "
        f"{initial_count}"
    )

    print(
        "Événements conservés après nettoyage : "
        f"{final_count}"
    )

    print(
        "Événements supprimés : "
        f"{initial_count - final_count}"
    )

    print(
        "Descriptions manquantes : "
        f"{missing_description} "
        f"({percentage(missing_description, final_count)} %)"
    )

    print(
        "Descriptions longues manquantes : "
        f"{missing_long_description} "
        f"({percentage(missing_long_description, final_count)} %)"
    )

    print(
        "Lieux manquants : "
        f"{missing_location} "
        f"({percentage(missing_location, final_count)} %)"
    )

    print(
        "Villes manquantes : "
        f"{missing_city} "
        f"({percentage(missing_city, final_count)} %)"
    )

    print(
        f"Période couverte : "
        f"{min_date} → {max_date}"
    )

    print("=" * 50 + "\n")


def fetch_events():
    if not API_KEY:
        raise ValueError(
            "OPENAGENDA_API_KEY manquante "
            "dans le fichier .env"
        )

    if not AGENDA_UID:
        raise ValueError(
            "OPENAGENDA_AGENDA_UID manquant "
            "dans le fichier .env"
        )

    now = datetime.now(
        timezone.utc
    )

    print(
        "Date de référence pour ce run "
        f"(recalculée à chaque exécution) : {now.isoformat()}"
    )

    past_start_date = (
        now - timedelta(days=90)
    )

    past_end_date = now

    future_start_date = now

    future_end_date = (
        now + timedelta(days=270)
    )

    print(
        "Récupération des événements passés..."
    )

    past_rows = fetch_event_period(
        start_date=past_start_date,
        end_date=past_end_date,
        max_events=MAX_PAST_EVENTS,
        sort_order="timings.desc",
    )

    print(
        f"{len(past_rows)} événement(s) passé(s) valide(s) récupéré(s)."
    )

    print(
        "Récupération des événements futurs..."
    )

    future_rows = fetch_event_period(
        start_date=future_start_date,
        end_date=future_end_date,
        max_events=MAX_FUTURE_EVENTS,
        sort_order="timings.asc",
    )

    print(
        f"{len(future_rows)} événement(s) futur(s) valide(s) récupéré(s)."
    )

    rows = past_rows + future_rows

    initial_count = len(rows)

    df, _ = clean_dataframe(
        rows
    )

    print_quality_report(
        df=df,
        initial_count=initial_count,
        past_count=len(past_rows),
        future_count=len(future_rows),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"{len(df)} événements sauvegardés."
    )

    print(
        f"Fichier créé : {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    fetch_events()