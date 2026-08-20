import ast
import os
import sys
import types
import warnings
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
    
    class _StubChatVertexAI:
        pass
        
    _fake_vertexai_module.ChatVertexAI = _StubChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai_module

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

load_dotenv()

# Ajustement ici : on remonte d'un niveau (vers la racine du projet)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_contexts(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value]

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def load_ragas_dataset(input_csv: str) -> Dataset:
    df = pd.read_csv(input_csv)
    
    # excluded_ids = {"Q02", "Q09", "Q10", "Q11", "Q12", "Q14"}
    # df = df[~df["id"].isin(excluded_ids)]
    
    required_columns = {"question", "answer", "contexts", "ground_truth"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {input_csv} : {missing_columns}. "
            "Vérifie que run_evaluation.py a bien été exécuté avant ce script."
        )

    df["contexts"] = df["contexts"].apply(parse_contexts)

    dataset_dict = {
        "question": df["question"].astype(str).tolist(),
        "answer": df["answer"].astype(str).tolist(),
        "contexts": df["contexts"].tolist(),
        "ground_truth": df["ground_truth"].astype(str).tolist(),
    }

    return Dataset.from_dict(dataset_dict)


def evaluate_with_ragas(input_csv: str, output_csv: str) -> None:
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY absente du fichier .env.")

    print("1. Chargement des données d'évaluation...")
    dataset = load_ragas_dataset(input_csv)
    print(f"   {len(dataset)} questions chargées depuis {input_csv}.")

    print("2. Initialisation du LLM juge et des embeddings locaux...")

    judge_llm = ChatOpenAI(
        model="mistral-small-latest",
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        temperature=0.0,
        max_retries=10,
        timeout=600.0,
    )

    judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    print("3. Lancement de l'évaluation Ragas...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, max_retries=10, timeout=600),
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS AGRÉGÉS")
    print("=" * 60)
    print(result)

    results_df = result.to_pandas()
    results_df.to_csv(output_csv, index=False)

    print(f"\n4. Résultats détaillés sauvegardés dans : {output_csv}")


if __name__ == "__main__":
    input_path = os.path.join(BASE_DIR, "data", "rag_output_complet3.csv")
    output_path = os.path.join(BASE_DIR, "data", "resultats_evaluation_ragas3.csv")

    evaluate_with_ragas(input_path, output_path)