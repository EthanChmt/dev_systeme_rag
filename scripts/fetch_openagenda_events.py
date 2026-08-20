import html
import re
from pathlib import Path

import pandas as pd
import requests

OUTPUT_PATH = Path("data/events_openagenda.csv")

def clean_text(value):
    if value is None:
        return ""
    
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
        
    if isinstance(value, dict):
        value = value.get("fr") or next(iter(value.values()), "")
        
    value = str(value)
    
    if value.lower() == "nan":
        return ""
        
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    
    return value.strip()

def fetch_and_clean_data():
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
    
    valid_rows = []
    limit = 100
    offset = 0
    max_events = 500

    print(f"Récupération de {max_events} événements passés via l'API publique Opendatasoft...")

    now = pd.Timestamp.now('UTC')
    one_year_ago = now - pd.Timedelta(days=365)

    while len(valid_rows) < max_events:
        params = {
            "refine": "location_region:Provence-Alpes-Côte d'Azur",
            "limit": limit,
            "offset": offset
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        payload = response.json()
        batch = payload.get("results", [])

        if not batch:
            break

        for event in batch:
            if len(valid_rows) >= max_events:
                break
            
            uid = event.get("uid")
            title = clean_text(event.get("title_fr"))
            begin_raw = event.get("firstdate_begin")
            
            if not title or not uid or not begin_raw:
                continue
                
            try:
                begin_dt = pd.to_datetime(begin_raw, utc=True)
                if not (one_year_ago <= begin_dt <= now):
                    continue
            except Exception:
                continue
                
            description = clean_text(event.get("description_fr"))
            long_description = clean_text(event.get("longdescription_fr"))
            conditions = clean_text(event.get("conditions_fr"))
            keywords = clean_text(event.get("keywords_fr"))
            end_raw = event.get("lastdate_end")
            location_name = clean_text(event.get("location_name"))
            address = clean_text(event.get("location_address"))
            city = clean_text(event.get("location_city"))
            department = clean_text(event.get("location_department"))
            region = clean_text(event.get("location_region"))
            country = clean_text(event.get("country_fr"))
            slug = event.get("slug")
            
            coords = event.get("location_coordinates")
            lat = coords.get("lat") if isinstance(coords, dict) else None
            lon = coords.get("lon") if isinstance(coords, dict) else None

            text_for_embedding = "\n".join([
                f"Titre : {title}",
                f"Description courte : {description}",
                f"Description longue : {long_description}",
                f"Conditions : {conditions}",
                f"Mots-clés : {keywords}",
                f"Lieu : {location_name}, {address}, {city}, {department}, {region}, {country}",
                f"Dates : {begin_raw} - {end_raw}"
            ])

            row = {
                "uid": uid,
                "slug": slug,
                "title": title,
                "description": description,
                "long_description": long_description,
                "conditions": conditions,
                "keywords": keywords,
                "begin": begin_raw,
                "end": end_raw,
                "location_name": location_name,
                "address": address,
                "city": city,
                "department": department,
                "region": region,
                "country": country,
                "latitude": lat,
                "longitude": lon,
                "text_for_embedding": clean_text(text_for_embedding),
            }
            valid_rows.append(row)

        offset += limit
        if offset >= payload.get("total_count", limit):
            break

    df = pd.DataFrame(valid_rows)
    
    if df.empty:
        raise ValueError("Le DataFrame est vide. L'API n'a rien renvoyé de valide.")

    df = df.drop_duplicates(subset=["uid"])
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce", utc=True)
    df["end"] = pd.to_datetime(df["end"], errors="coerce", utc=True)
    
    df = df.sort_values(by="begin", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"SUCCÈS : {len(df)} événements propres et vérifiés sauvegardés dans {OUTPUT_PATH}")

if __name__ == "__main__":
    fetch_and_clean_data()