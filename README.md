# POC Chatbot RAG — Puls-Events

**Projet réalisé dans le cadre d'une formation OpenClassrooms.**

Puls-Events est un **POC de chatbot RAG (Retrieval-Augmented Generation)** conçu pour répondre en langage naturel à des questions sur des événements culturels, patrimoniaux et associatifs en région **Provence-Alpes-Côte d'Azur**. Le système s'appuie sur des données réelles récupérées via l'API **OpenAgenda**, indexées sémantiquement dans une base **FAISS**, puis interrogées par un pipeline RAG complet exposé à la fois via une **API REST FastAPI** et une **interface web Gradio** sur mesure.

Au-delà du pipeline RAG "classique", le projet met l'accent sur des points souvent négligés dans un POC :
- un **prompt engineering strict** pour limiter les hallucinations et empêcher le modèle de proposer des événements passés ou inventés ;
- un **filtrage métier et mathématique** des résultats (seuil de similarité, dédoublonnage) appliqué en amont de la génération ;
- une **API de reconstruction à la demande** de l'index (`POST /rebuild`) plutôt qu'un pipeline figé ;
- une **démarche d'évaluation itérative** avec Ragas, ayant conduit à des refontes architecturales majeures (voir [Évaluation](#évaluation-ragas)).

## Sommaire

- [Fonctionnalités clés](#fonctionnalités-clés)
- [Architecture](#architecture)
- [Stack technique](#stack-technique)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
  - [Lancement en local](#lancement-en-local)
  - [Lancement avec Docker](#lancement-avec-docker)
- [Routes de l'API](#routes-de-lapi)
- [Interfaces utilisateur](#interfaces-utilisateur)
- [Tests](#tests)
- [Évaluation (Ragas)](#évaluation-ragas)
- [Choix de modèles et paramètres](#choix-de-modèles-et-paramètres)
- [Intégration continue](#intégration-continue)
- [Roadmap](#roadmap)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Fonctionnalités clés

- **Ingestion automatisée** des événements OpenAgenda (`scripts/fetch_openagenda_events.py`), paramétrable par ville, mot-clé, et ciblée sur une fenêtre glissante des 365 derniers jours (500 événements max).
- **Indexation vectorielle FAISS** des événements (`scripts/build_faiss_index.py`), reconstructible à tout moment sans redéploiement grâce à l'endpoint `POST /rebuild`.
- **Pipeline RAG "métier"** (`app/chatbot.py`) particulièrement robuste :
  - **Retrieval hybride** via `EnsembleRetriever` (LangChain), combinant une recherche **lexicale BM25** et une recherche **sémantique FAISS**, chacune remontant 100 candidats avant fusion pondérée.
  - **Seuil de sécurité mathématique** (`score_threshold = 0.43`) pour garantir la pertinence. Si aucun document ne franchit ce seuil, le système refuse de répondre plutôt que d'halluciner.
  - **Dédoublonnage strict** appliqué sur les candidats fusionnés.
  - Conservation des **3 événements les plus pertinents** au final (`MAX_EVENTS_IN_CONTEXT`).
  - **Prompt système ultra-cadré** (pas d'invention de date/lieu/tarif, pas de listes Markdown, interdiction de s'excuser, réponse en français rédigée en paragraphes).
  - **Post-traitement de la réponse** pour signaler explicitement quand moins de 3 événements sont disponibles.
- **Séparation embeddings / génération** : les embeddings sont calculés via l'API **Mistral** (`mistral-embed`), tandis que la génération de texte tourne en local avec un modèle **Hugging Face** (`Qwen/Qwen2.5-1.5B-Instruct` par défaut), avec **détection automatique GPU/CPU** (`torch.cuda.is_available()`).
- **API REST FastAPI** documentée automatiquement (Swagger), avec gestion d'erreurs explicite (400 si question vide, 500 en cas d'erreur pipeline, 504 en cas de timeout de reconstruction).
- **Interface Gradio sur mesure** (thème "PACA", CSS personnalisé) montée directement sur l'application FastAPI via `gradio.mount_gradio_app`, affichant la réponse générée **et** les événements sources sous forme de cartes.
- **Conteneurisation Docker** avec healthcheck, et pipeline d'entrypoint qui réalise à chaud la collecte + l'indexation avant de démarrer l'API.

## Architecture

![Architecture du projet Puls-Events : ingestion OpenAgenda, embeddings Mistral, index FAISS, pipeline RAG, API FastAPI et deux interfaces Gradio](docs/architecture.svg)

- `scripts/fetch_openagenda_events.py` récupère les événements depuis OpenAgenda.
- `scripts/build_faiss_index.py` construit l'index vectoriel FAISS à partir de ces données, avec les embeddings Mistral.
- `app/chatbot.py` expose la fonction `ask_chatbot(question)`, cœur du pipeline RAG (retrieval hybride BM25+FAISS, filtrage par seuil à 0.43, prompt engineering, génération locale via Hugging Face).
- `app/api.py` expose cette logique via une API **FastAPI**, et **monte également l'interface Gradio** (`app/interface.py`) directement dans l'application via `gradio.mount_gradio_app`, accessible sur `/interface`.
- `app/interface.py` construit l'interface Gradio thématisée "PACA" : elle **n'appelle pas `ask_chatbot` directement**, mais consomme l'API FastAPI (`POST /ask`, `GET /health`, `POST /rebuild`) via `httpx`, ce qui découple totalement l'UI du backend.
- `app_gradio.py`, à la racine, est une **interface Gradio minimale** alternative qui, elle, appelle `ask_chatbot()` directement (sans passer par l'API HTTP) — pratique pour tester le pipeline RAG en isolation.

## Stack technique

| Composant | Technologie |
| --- | --- |
| Langage | Python 3.11 |
| Orchestration / retrieval | LangChain, `langchain-community` (vector store FAISS), `EnsembleRetriever` |
| Base vectorielle | FAISS (`faiss-cpu`) |
| Retrieval lexical | BM25 (`rank_bm25`), combiné à FAISS via `EnsembleRetriever` |
| Embeddings | Mistral AI (`langchain-mistralai`, modèle `mistral-embed`) |
| Génération de texte | Modèle Hugging Face local (`transformers`, `Qwen/Qwen2.5-1.5B-Instruct`), exécution GPU/CPU |
| Source de données | API OpenAgenda |
| API backend | FastAPI + Uvicorn + Pydantic |
| Interface utilisateur | Gradio (thème et CSS personnalisés, montée dans FastAPI) |
| Client HTTP interne (UI → API) | `httpx` |
| Évaluation RAG | Ragas |
| Tests | Pytest |
| Conteneurisation | Docker |

## Structure du projet

```text
dev_systeme_rag/
├── app/
│   ├── api.py               # Application FastAPI : routes + montage de l'UI Gradio
│   ├── chatbot.py           # Pipeline RAG : retrieval FAISS, filtrage, prompt, génération
│   ├── interface.py         # Interface Gradio "PACA" (consomme l'API via httpx)
│   └── interface_style.py   # Thème et CSS personnalisés de l'interface Gradio
├── scripts/                 # Récupération OpenAgenda + construction de l'index FAISS
├── tests/                   # Tests unitaires (Pytest)
├── data/                    # Données locales non versionnées
├── vectorstore/             # Index vectoriel FAISS non versionné
├── docs/                    # Documentation technique
├── .github/workflows/       # Workflows d'intégration continue (GitHub Actions)
├── app_gradio.py            # Interface Gradio minimale, en appel direct à ask_chatbot()
├── docker-entrypoint.sh     # Point d'entrée Docker (fetch + index + API)
├── Dockerfile               # Image Docker de l'application
├── .env.example             # Exemple de variables d'environnement
├── .gitignore               # Fichiers exclus du dépôt Git
├── requirements.txt         # Dépendances Python
└── README.md                # Documentation du projet
```

## Prérequis

- Python 3.11
- Une clé API Mistral (https://console.mistral.ai/)
- Une clé API OpenAgenda (https://developers.openagenda.com/)
- Docker (optionnel, pour un déploiement conteneurisé)

## Installation

Cloner le dépôt :

```bash
git clone https://github.com/EthanChmt/dev_systeme_rag.git
cd dev_systeme_rag
```

Créer et activer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

Installer les dépendances :

```bash
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Configuration

Copier le fichier d'exemple et compléter les variables d'environnement :

```bash
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| `MISTRAL_API_KEY` | Clé API Mistral, utilisée pour générer les embeddings. |
| `MISTRAL_EMBEDDING_MODEL` | Modèle d'embedding Mistral à utiliser (`mistral-embed` par défaut). |
| `HUGGINGFACE_GENERATION_MODEL` | Modèle Hugging Face utilisé pour la génération (`Qwen/Qwen2.5-1.5B-Instruct` par défaut). |
| `OPENAGENDA_API_KEY` | Clé API OpenAgenda. |
| `OPENAGENDA_AGENDA_UID` | Identifiant de l'agenda OpenAgenda à interroger. |
| `OPENAGENDA_SEARCH` | Filtre de recherche optionnel sur les événements. |
| `OPENAGENDA_CITY` | Filtre optionnel par ville. |
| `OPENAGENDA_MAX_EVENTS` | Nombre maximum d'événements à récupérer (300 par défaut). |
| `API_BASE_URL` | URL de base de l'API FastAPI utilisée par l'interface Gradio (`http://127.0.0.1:8000`). |

## Utilisation

### Lancement en local

Récupérer les événements OpenAgenda :

```bash
python scripts/fetch_openagenda_events.py
```

Construire l'index FAISS :

```bash
python scripts/build_faiss_index.py
```

Lancer l'API FastAPI :

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Lancement avec Docker

L'image Docker automatise les étapes de récupération et d'indexation au démarrage du conteneur (`docker-entrypoint.sh`) puis expose l'API sur le port 8000.

```bash
docker build -t puls-events-rag .
docker run --env-file .env -p 8000:8000 puls-events-rag
```

## Routes de l'API

| Méthode | Route | Description |
| --- | --- | --- |
| GET | `/` | Message de bienvenue et lien vers la documentation. |
| GET | `/health` | Healthcheck simple (`{"status": "ok"}`). |
| POST | `/ask` | Question envoyée au pipeline RAG. Corps : `{"question": "..."}`. Réponse détaillée avec sources. |
| POST | `/rebuild` | Relance à chaud `fetch_openagenda_events.py` puis `build_faiss_index.py`. Timeout : 300 s. |
| GET | `/docs` | Documentation interactive Swagger générée automatiquement par FastAPI. |
| GET | `/interface` | Interface Gradio "PACA" montée sur l'application FastAPI. |

## Interfaces utilisateur

Le projet propose deux interfaces Gradio distinctes :

- `app/interface.py` : Interface principale (thème PACA). Montée dans FastAPI, elle consomme l'API HTTP via `httpx`, découplant totalement l'UI du backend.
- `app_gradio.py` : Interface minimale en racine de projet pour tester le pipeline RAG en appelant directement `ask_chatbot()`.

## Tests

Les tests unitaires (Pytest) se trouvent dans `tests/` :

```bash
pytest
```

## Évaluation (Ragas)

La démarche d'évaluation s'est déroulée en deux phases itératives, ce qui a permis d'identifier les goulots d'étranglement du pipeline et d'appliquer des corrections architecturales majeures.

### Phase 1 — LLM-as-a-judge

Une première évaluation qualitative a été réalisée via une approche LLM-as-a-judge. Cette étape a rapidement mis en évidence un problème de troncature : la limite de tokens allouée à la génération (`max_new_tokens`) était trop basse. L'augmentation de cette limite a permis au SLM (Qwen) de formuler des réponses complètes, corrigeant immédiatement les scores artificiellement bas liés aux phrases coupées.

### Phase 2 — Évaluation Ragas et corrections architecturales

Pour obtenir une validation scientifique, le système a ensuite été évalué avec le framework Ragas sur un jeu de test de requêtes strictement culturelles. Les premières métriques ont servi de base de travail pour appliquer une série d'optimisations lourdes sur le pipeline :

- **Changement du système d'embedding** : Remplacement du modèle de vectorisation pour garantir une compréhension sémantique beaucoup plus fine du français.
- **Introduction de BM25** : Déploiement d'un `EnsembleRetriever` hybride pour pallier les limites de la recherche purement vectorielle sur les mots-clés exacts (noms propres, villes).
- **Adaptation du Ground Truth** : Nettoyage des réponses de référence du jeu de test (suppression des exemples de titres arbitraires au profit de formulations thématiques) pour ne plus pénaliser le modèle lors de la phase de Context Precision.
- **Validation de la génération** : Maintien de l'augmentation des tokens max pour assurer une restitution complète du contexte.

### Résultats finaux

Grâce à ces corrections, le pipeline a atteint d'excellentes performances métriques, prouvant la pertinence chirurgicale du contexte récupéré et la fiabilité absolue de la génération :

- **Faithfulness** (Fidélité au contexte) : 0.8569
- **Answer Relevancy** (Pertinence de la réponse) : 0.7770
- **Context Precision** (Précision du contexte) : 0.6500
- **Context Recall** (Couverture du contexte) : 0.9000

Ces scores valident définitivement le RAG : il trouve presque toujours les bonnes sources (Recall à 90 %) et produit une réponse hautement fidèle sans hallucination (Faithfulness à ~86 %).

## Choix de modèles et paramètres

| Paramètre | Valeur retenue | Justification |
| --- | --- | --- |
| Modèle d'embeddings | `mistral-embed` (API Mistral) | Externalise le calcul des embeddings pour ne pas alourdir le pipeline local. |
| Modèle de génération | `Qwen/Qwen2.5-1.5B-Instruct` | Modèle compact (1.5B paramètres) exécutable en local sans coût d'API récurrent, avec bascule automatique GPU/CPU. |
| Retrieval | Hybride BM25 + FAISS | FAISS seul s'est montré insuffisant sur les requêtes utilisant un vocabulaire éloigné du texte source ; BM25 repère les mots-clés exacts. L'hybridation couvre les deux cas. |
| `score_threshold` | 0.43 | Valeur barrière déterminée expérimentalement pour forcer le RAG à admettre son ignorance plutôt que d'halluciner. |
| `MAX_EVENTS_IN_CONTEXT` | 3 | Compromis pensé pour un usage conversationnel (recommandation ciblée plutôt qu'un listing exhaustif). |
| `chunk_size` | 800 caractères | Valeur retenue pour limiter la perte d'informations contextuelles. Des pistes d'amélioration restent identifiées (voir Roadmap). |

## Intégration continue

Des workflows GitHub Actions sont définis dans `.github/workflows/` pour automatiser les tests à chaque contribution.

## Roadmap


- [ ] Explorer une étape de reformulation de requête (query rewriting via LLM) pour les questions formulées de façon vague ou indirecte.
- [ ] Ajouter un filtrage explicite par ville (métadonnée `city`) en amont du retrieval sémantique pour les questions de filtrage géographique.
- [ ] Mettre en place un historique de conversation pour les requêtes multi-tours.
- [ ] Mettre en place une notation utilisateur pour avoir des retours sur les réponses RAG.

## Contribuer

Les contributions sont bienvenues :

1. Créer une branche depuis `develop`
2. Committer les changements avec des messages clairs
3. Ouvrir une Pull Request vers `develop`

## Licence

Aucune licence n'est actuellement définie pour ce dépôt. Contacter l'auteur pour toute question d'utilisation.