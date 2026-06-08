# User Engagement & Health Score Platform

Full-stack deployment of the engagement analytics backend with an interactive dashboard frontend.

---

## Stack

| Layer     | Tech |
|-----------|------|
| Backend   | Python · Flask · Pandas · MongoDB · OpenAI |
| Frontend  | Vanilla JS · Tailwind CDN · Chart.js |
| Infra     | Docker · docker-compose |

---

## Quick Start

### Docker (recommended)

```bash
# Optional: add your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

docker-compose up --build
```

Open **http://localhost:5000**

> Without an OpenAI key the platform falls back to built-in rule-based recommendations automatically.

### Local Python

```bash
# MongoDB must be running on localhost:27017
pip install -r requirements.txt
MONGO_URI=mongodb://localhost:27017/ python app.py
```

---

## Dashboard walkthrough

| Feature | Where |
|---------|-------|
| Simulate a new user (generates events, scores, recs) | Sidebar → **Ingest & Analyse** |
| Load an existing user | Sidebar → **Load User** |
| Quick-switch between users | Sidebar → **Known Users** list |
| Filter at-risk / churning users | Sidebar → **At-Risk Users** |
| Refresh AI recommendations | Top-right → **Refresh AI Recs** |
| Score breakdown | Tab → **Overview** |
| Raw engagement metrics | Tab → **Metrics** |
| AI / rule-based recommendations | Tab → **Recommendations** |
| 30-day health score trend | Tab → **Score Trend** |

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/events`                  | Ingest user events |
| GET    | `/health-score/<user_id>`  | Compute / retrieve health score |
| GET    | `/recommendations/<user_id>` | Generate AI recommendations |
| GET    | `/dashboard/<user_id>`     | Full dashboard data |
| GET    | `/users/gaps`              | At-risk users with engagement gaps |
| GET    | `/users`                   | List all known user IDs |
| GET    | `/health`                  | Service health check |

---

## Project structure

```
engagement_app/
├── app.py               # Flask backend (all original logic + frontend serving)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── templates/
    └── index.html       # Interactive dashboard (Chart.js + Tailwind)
```
