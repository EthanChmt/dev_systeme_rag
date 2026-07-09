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
CITY = os.getenv("OPENAGENDA_CITY", "Paris")
SEARCH = os.getenv("OPENAGENDA_SEARCH", "")
MAX_EVENTS = int(os.getenv("OPENAGENDA_MAX_EVENTS", "300"))

OUTPUT_PATH = Path("data/events_openagenda.csv")


def clean_text(value):
    if value is None:
        return ""

    if isinstance(value, dict):
        value = value.get("fr") or next(iter(value.values()), "")

    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)

    value = str(value)
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def first_timing(event):
    timings = event.get("timings") or []
    if not timings:
        return "", ""

    first = timings[0]
    return first.get("begin", ""), first.get("end", "")


def normalize_event(event):
    location = event.get("location") or {}
    begin, end = first_timing(event)

    title = clean_text(event.get("title"))
    description = clean_text(event.get("description"))
    long_description = clean_text(event.get("longDescription"))
    conditions = clean_text(event.get("conditions"))
    keywords = clean_text(event.get("keywords"))

    location_name = clean_text(location.get("name"))
    address = clean_text(location.get("address"))
    city = clean_text(location.get("city") or location.get("adminLevel4"))
    department = clean_text(location.get("department") or location.get("adminLevel2"))
    region = clean_text(location.get("region") or location.get("adminLevel1"))
    country = clean_text(location.get("country"))

    text_for_embedding = "\n".join(
        [
            f"Titre : {title}",
            f"Description courte : {description}",
            f"Description longue : {long_description}",
            f"Conditions : {conditions}",
            f"Mots-clés : {keywords}",
            f"Lieu : {location_name}, {address}, {city}, {department}, {region}, {country}",
            f"Dates : {begin} - {end}",
        ]
    )

    return {
        "uid": event.get("uid"),
        "slug": event.get("slug"),
        "title": title,
        "description": description,
        "long_description": long_description,
        "conditions": conditions,
        "keywords": keywords,
        "begin": begin,
        "end": end,
        "location_name": location_name,
        "address": address,
        "city": city,
        "department": department,
        "region": region,
        "country": country,
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "text_for_embedding": clean_text(text_for_embedding),
    }


def fetch_events():
    if not API_KEY:
        raise ValueError("OPENAGENDA_API_KEY manquante dans le fichier .env")

    if not AGENDA_UID:
        raise ValueError("OPENAGENDA_AGENDA_UID manquant dans le fichier .env")

    today = datetime.now(timezone.utc)
    start_date = today - timedelta(days=365)
    end_date = today + timedelta(days=365)

    url = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"
    headers = {"key": API_KEY}

    events = []
    after = None

    while len(events) < MAX_EVENTS:
        params = [
            ("size", min(300, MAX_EVENTS - len(events))),
            ("detailed", "1"),
            ("monolingual", "fr"),
            ("sort", "timings.asc"),
            ("timings[gte]", start_date.isoformat().replace("+00:00", "Z")),
            ("timings[lte]", end_date.isoformat().replace("+00:00", "Z")),
        ]

        if CITY:
            params.append(("adminLevel4[]", CITY))

        if SEARCH:
            params.append(("search", SEARCH))
            params.append(("threshold", "auto"))

        if after:
            for value in after:
                params.append(("after[]", value))

        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()

        payload = response.json()
        batch = payload.get("events", [])

        if not batch:
            break

        events.extend(batch)
        after = payload.get("after")

        if not after:
            break

    rows = [normalize_event(event) for event in events]
    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"{len(df)} événements récupérés.")
    print(f"Fichier créé : {OUTPUT_PATH}")


if __name__ == "__main__":
    fetch_events()
