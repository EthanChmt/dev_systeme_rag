from pathlib import Path
import pandas as pd


def test_openagenda_csv_exists():
    path = Path("data/events_openagenda.csv")
    assert path.exists(), "Le fichier data/events_openagenda.csv est absent."


def test_openagenda_csv_is_not_empty():
    df = pd.read_csv("data/events_openagenda.csv")
    assert len(df) > 0, "Le fichier CSV ne contient aucun événement."


def test_openagenda_csv_has_expected_columns():
    df = pd.read_csv("data/events_openagenda.csv")

    expected_columns = {
        "uid",
        "title",
        "description",
        "begin",
        "end",
        "location_name",
        "city",
        "text_for_embedding",
    }

    missing_columns = expected_columns - set(df.columns)

    assert not missing_columns, f"Colonnes manquantes : {missing_columns}"


def test_embedding_text_is_usable():
    df = pd.read_csv("data/events_openagenda.csv")

    assert df["text_for_embedding"].notna().any()
    assert df["text_for_embedding"].str.len().max() > 50