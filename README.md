---
title: Puls Events RAG
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.11"
app_file: app_gradio.py
pinned: false
---



# POC Chatbot RAG — Puls-Events

## Objectif

Ce projet est un POC de chatbot RAG pour Puls-Events.  
Il vise à démontrer la faisabilité technique d'un système capable de répondre à des questions utilisateurs sur des événements culturels à partir de données récupérées, nettoyées, vectorisées et interrogées via une base FAISS.

Le système utilisera notamment :

- Python ;
- LangChain ;
- FAISS ;
- Mistral ;
- FastAPI ;
- OpenAgenda API ;
- Ragas pour l'évaluation.

## Structure du projet

```text
dev_sys_RAG/
├── app/              # Code applicatif et API
├── scripts/          # Scripts de récupération, nettoyage et vectorisation
├── tests/            # Tests unitaires
├── data/             # Données locales non versionnées
├── vectorstore/      # Index vectoriel FAISS non versionné
├── docs/             # Documentation technique
├── .env.example      # Exemple de variables d'environnement
├── .gitignore        # Fichiers exclus du dépôt Git
├── requirements.txt  # Dépendances Python
└── README.md         # Documentation du projet
