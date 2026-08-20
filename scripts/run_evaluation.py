import pandas as pd
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.chatbot import ask_chatbot

def generer_donnees_evaluation(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    
    reponses_generees = []
    contextes_recuperes = []
    total = len(df)
    
    for index, row in df.iterrows():
        question = row['question']
        print(f"Traitement {index + 1}/{total} : {question}")
        
        try:
            result = ask_chatbot(question)
            
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            texts = []
            for s in sources:
                texte_source = (
                    f"Titre : {s.get('title', '')}\n"
                    f"Lieu : {s.get('location_name', '')} - {s.get('city', '')}\n"
                    f"Dates : {s.get('begin', '')} au {s.get('end', '')}\n"
                    f"Adresse : {s.get('address', '')}\n"
                    f"Description : {s.get('content', '')}"
                )
                texts.append(texte_source.strip())
                
            reponses_generees.append(answer)
            contextes_recuperes.append(texts)
            
        except Exception as e:
            print(f"Erreur sur la question {index + 1} : {e}")
            reponses_generees.append("Erreur")
            contextes_recuperes.append([])
            
    df['answer'] = reponses_generees
    df['contexts'] = contextes_recuperes
    
    if 'reponse_attendue' in df.columns:
        df = df.rename(columns={'reponse_attendue': 'ground_truth'})
        
    df.to_csv(output_csv, index=False)
    print(f"\nFichier final sauvegardé : {output_csv}")

if __name__ == "__main__":
    input_path = os.path.join(BASE_DIR, "data", "questions_test_rag_v3.csv")
    output_path = os.path.join(BASE_DIR, "data", "rag_output_complet3.csv")
    generer_donnees_evaluation(input_path, output_path)