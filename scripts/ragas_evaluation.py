import ast
import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_contexts(val) -> str:
    if pd.isna(val): return ""
    if isinstance(val, str) and val.startswith('['):
        try: val = ast.literal_eval(val)
        except (ValueError, SyntaxError): pass
    if isinstance(val, list): return "\n".join(str(item) for item in val)
    return str(val)

def extract_score(text: str) -> int:
    match = re.search(r'\d+', text)
    if match: return max(1, min(int(match.group()), 5))
    return 1

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=30))
def invoke_with_retry(chain, params):
    return chain.invoke(params)

def evaluate_rag_custom(input_csv: str, output_csv: str, log_file_path: str) -> None:
    df = pd.read_csv(input_csv)
    df['contexts'] = df['contexts'].apply(parse_contexts)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key: raise ValueError("MISTRAL_API_KEY absente.")

    llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=api_key, temperature=0.0)

    prompt_fidelite = ChatPromptTemplate.from_messages([
        ("system", "Évalue la fidélité de 1 à 5. 5=Parfait, 4=Léger détail inventé, 3=Moyen, 2=Mauvais, 1=Hors contexte. Réponds UNIQUEMENT par le chiffre."),
        ("user", "Contexte :\n{context}\n\nRéponse :\n{answer}")
    ])

    prompt_pertinence = ChatPromptTemplate.from_messages([
        ("system", "Évalue la pertinence de 1 à 5. 5=Parfait, 4=Un peu bavard, 3=Partiel/Coupé, 2=Esquive, 1=Hors sujet. Réponds UNIQUEMENT par le chiffre."),
        ("user", "Question :\n{question}\n\nRéponse :\n{answer}")
    ])

    chain_fidelite = prompt_fidelite | llm
    chain_pertinence = prompt_pertinence | llm

    scores_fidelite = []
    scores_pertinence = []

    log_file = open(log_file_path, "w", encoding="utf-8")

    for index, row in df.iterrows():
        print(f"Évaluation de la question {index + 1}/{len(df)}...")
        answer = str(row.get('answer', ''))
        
        try:
            res_fid = invoke_with_retry(chain_fidelite, {"context": str(row.get('contexts', '')), "answer": answer})
            scores_fidelite.append(extract_score(res_fid.content))
        except Exception as e:
            msg = f"Q{index+1} - Erreur fidélité: {e}\n"
            print(f"  {msg}")
            log_file.write(msg)
            scores_fidelite.append(1)
            
        time.sleep(2)

        try:
            res_pert = invoke_with_retry(chain_pertinence, {"question": str(row.get('question', '')), "answer": answer})
            scores_pertinence.append(extract_score(res_pert.content))
        except Exception as e:
            msg = f"Q{index+1} - Erreur pertinence: {e}\n"
            print(f"  {msg}")
            log_file.write(msg)
            scores_pertinence.append(1)
            
        time.sleep(2)

    log_file.close()

    df['score_fidelite'] = scores_fidelite
    df['score_pertinence'] = scores_pertinence
    df.to_csv(output_csv, index=False)
    print(f"\nTerminé ! Fichier sauvegardé : {output_csv}")
    print(f"Vérifiez le fichier {os.path.basename(log_file_path)} en cas d'anomalies.")

if __name__ == "__main__":
    input_path = os.path.join(BASE_DIR, "data", "rag_output_complet.csv")
    output_path = os.path.join(BASE_DIR, "data", "resultats_evaluation.csv")
    log_path = os.path.join(BASE_DIR, "data", "mistral_errors.log")
    
    evaluate_rag_custom(input_path, output_path, log_path)