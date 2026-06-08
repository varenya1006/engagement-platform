"""
User Engagement & Health Score Platform
=========================================
Full-stack deployment: original backend (MongoDB, health-score engine,
OpenAI recommendations) + interactive dashboard frontend served by Flask.

Tech Stack: Python | Flask | Pandas | MongoDB | OpenAI | Docker | REST API
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
from pymongo.collection import Collection
import openai

app = Flask(__name__)
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────
MONGO_URI       = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── MongoDB ────────────────────────────────────────────────────────────────────
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo_client["engagement_platform"]

users_col:           Collection = db["users"]
events_col:          Collection = db["events"]
health_scores_col:   Collection = db["health_scores"]
recommendations_col: Collection = db["recommendations"]

def _ensure_indexes():
    try:
        events_col.create_index([("user_id", ASCENDING), ("timestamp", DESCENDING)])
        health_scores_col.create_index([("user_id", ASCENDING), ("computed_at", DESCENDING)])
    except Exception as e:
        app.logger.warning(f"Index creation skipped: {e}")

with app.app_context():
    _ensure_indexes()


# ═══════════════════════════════════════════════════════════════════════════════
# Health Score Engine
# ═══════════════════════════════════════════════════════════════════════════════

class HealthScoreEngine:
    WEIGHTS = {
        "activity_frequency": 0.30,
        "session_depth":      0.25,
        "feature_adoption":   0.25,
        "recency":            0.20,
    }

    @staticmethod
    def compute_metrics(user_events: List[Dict]) -> Dict:
        if not user_events:
            return {
                "total_sessions": 0, "total_events": 0,
                "avg_session_duration": 0, "unique_features": 0,
                "days_since_last_activity": 999, "events_per_session": 0,
            }

        df = pd.DataFrame(user_events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        df["time_diff"]  = df["timestamp"].diff().dt.total_seconds()
        df["new_session"] = df["time_diff"] > 1800
        df["session_id"]  = df["new_session"].cumsum()

        sessions        = df.groupby("session_id")
        total_sessions  = int(df["session_id"].nunique())
        total_events    = len(df)
        avg_session_dur = float(sessions["time_diff"].sum().mean()) if total_sessions > 0 else 0
        unique_features = int(df["feature"].nunique()) if "feature" in df.columns else 0
        last_activity   = df["timestamp"].max()
        days_since      = (datetime.utcnow() - last_activity).days
        eps             = total_events / total_sessions if total_sessions > 0 else 0

        return {
            "total_sessions":           total_sessions,
            "total_events":             total_events,
            "avg_session_duration":     round(avg_session_dur, 2),
            "unique_features":          unique_features,
            "days_since_last_activity": int(days_since),
            "events_per_session":       round(eps, 2),
        }

    @classmethod
    def compute_health_score(cls, metrics: Dict) -> Dict:
        activity_score = min(100, metrics["total_sessions"] * 10)
        session_score  = min(100, metrics["avg_session_duration"] / 60 * 20)
        feature_score  = min(100, metrics["unique_features"] * 25)
        recency_score  = max(0, 100 - metrics["days_since_last_activity"] * 10)

        composite = (
            cls.WEIGHTS["activity_frequency"] * activity_score +
            cls.WEIGHTS["session_depth"]      * session_score  +
            cls.WEIGHTS["feature_adoption"]   * feature_score  +
            cls.WEIGHTS["recency"]            * recency_score
        )

        churn_risk = max(0, min(100, 100 - composite))
        risk_level = "high" if churn_risk > 70 else "medium" if churn_risk > 40 else "low"

        gaps = []
        if metrics["days_since_last_activity"] > 7:  gaps.append("inactive_week")
        if metrics["unique_features"] < 2:            gaps.append("low_feature_adoption")
        if metrics["total_sessions"] < 3:             gaps.append("low_session_count")

        return {
            "composite_score":   round(composite, 1),
            "component_scores": {
                "activity_frequency": round(activity_score, 1),
                "session_depth":      round(session_score, 1),
                "feature_adoption":   round(feature_score, 1),
                "recency":            round(recency_score, 1),
            },
            "churn_risk":        {"score": round(churn_risk, 1), "level": risk_level},
            "engagement_gaps":   gaps,
            "weights_used":      cls.WEIGHTS,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Recommendation Microservice
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_based_recommendations(gaps: List[str], churn_level: str) -> List[Dict]:
    """Fallback recommendations when OpenAI key is absent."""
    rec_map = {
        "inactive_week":        {"action": "Send a re-engagement email with personalised highlights", "expected_impact": "high"},
        "low_feature_adoption": {"action": "Trigger an in-app feature tour for unused modules",      "expected_impact": "medium"},
        "low_session_count":    {"action": "Push a daily digest notification to build the habit",    "expected_impact": "medium"},
    }
    defaults = [
        {"action": "Offer a live onboarding session to increase product fluency",  "expected_impact": "high"},
        {"action": "Unlock a loyalty badge after next 3 sessions",                 "expected_impact": "medium"},
        {"action": "Send weekly progress report to reinforce value perception",    "expected_impact": "medium"},
    ]
    recs = [rec_map[g] for g in gaps if g in rec_map]
    recs += [r for r in defaults if r not in recs]
    return recs[:3]


def generate_recommendations(user_id: str, health_data: Dict, metrics: Dict) -> Dict:
    gaps       = health_data.get("engagement_gaps", [])
    churn_level = health_data.get("churn_risk", {}).get("level", "unknown")

    if not openai_client:
        return {
            "user_id": user_id,
            "recommendations": _rule_based_recommendations(gaps, churn_level),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": "rule-based (no OpenAI key configured)",
        }

    prompt = f"""You are an engagement strategist. Based on the following user metrics, provide 3 actionable recommendations to improve user engagement.

User Metrics:
- Total Sessions: {metrics['total_sessions']}
- Avg Session Duration: {metrics['avg_session_duration']:.0f} seconds
- Unique Features Used: {metrics['unique_features']}
- Days Since Last Activity: {metrics['days_since_last_activity']}
- Health Score: {health_data['composite_score']}/100
- Churn Risk: {churn_level}
- Engagement Gaps: {', '.join(gaps) if gaps else 'None detected'}

Provide exactly 3 short, specific recommendations. Format as a JSON array of objects with "action" and "expected_impact" fields."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful engagement analytics assistant. Always respond with valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.5, max_tokens=400,
        )
        content = response.choices[0].message.content
        try:
            recommendations = json.loads(content)
        except json.JSONDecodeError:
            recommendations = [{"action": content, "expected_impact": "moderate"}]

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": "gpt-3.5-turbo",
        }
    except Exception as e:
        return {
            "user_id": user_id,
            "recommendations": _rule_based_recommendations(gaps, churn_level),
            "error": str(e),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": "rule-based (OpenAI error)",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: serialise ObjectId
# ═══════════════════════════════════════════════════════════════════════════════

def _clean(doc: Dict) -> Dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ═══════════════════════════════════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# REST API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/events", methods=["POST"])
def ingest_events():
    data     = request.get_json(force=True)
    user_id  = data.get("user_id")
    events   = data.get("events", [])
    if not user_id or not events:
        return jsonify({"error": "user_id and events are required"}), 400

    enriched = []
    for e in events:
        e["user_id"]    = user_id
        e["ingested_at"] = datetime.utcnow().isoformat() + "Z"
        e["event_id"]   = str(uuid.uuid4())
        enriched.append(e)

    try:
        events_col.insert_many(enriched)
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    return jsonify({"status": "events_ingested", "user_id": user_id, "event_count": len(enriched)}), 201


@app.route("/health-score/<user_id>", methods=["GET"])
def get_health_score(user_id: str):
    recalculate = request.args.get("recalculate", "false").lower() == "true"

    try:
        if not recalculate:
            cached = health_scores_col.find_one({"user_id": user_id}, sort=[("computed_at", DESCENDING)])
            if cached:
                return jsonify(_clean(cached)), 200

        user_events = list(events_col.find({"user_id": user_id}, {"_id": 0})
                           .sort("timestamp", DESCENDING).limit(1000))
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500

    metrics     = HealthScoreEngine.compute_metrics(user_events)
    health_data = HealthScoreEngine.compute_health_score(metrics)

    result = {
        "user_id":     user_id,
        "metrics":     metrics,
        "health_score": health_data,
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        health_scores_col.insert_one(result)
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    return jsonify(_clean(result)), 200


@app.route("/recommendations/<user_id>", methods=["GET"])
def get_recommendations(user_id: str):
    try:
        health_record = health_scores_col.find_one({"user_id": user_id}, sort=[("computed_at", DESCENDING)])
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    if not health_record:
        return jsonify({"error": "No health score found. Compute health score first."}), 404

    recs = generate_recommendations(user_id, health_record.get("health_score", {}), health_record.get("metrics", {}))
    recs["user_id"] = user_id
    try:
        recommendations_col.insert_one(recs)
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    return jsonify(_clean(recs)), 200


@app.route("/dashboard/<user_id>", methods=["GET"])
def get_dashboard(user_id: str):
    health_record       = health_scores_col.find_one({"user_id": user_id}, sort=[("computed_at", DESCENDING)])
    latest_rec          = recommendations_col.find_one({"user_id": user_id}, sort=[("generated_at", DESCENDING)])
    thirty_days_ago     = datetime.utcnow() - timedelta(days=30)
    trend_records       = list(health_scores_col.find(
        {"user_id": user_id, "computed_at": {"$gte": thirty_days_ago.isoformat()}},
        {"_id": 0, "computed_at": 1, "health_score.composite_score": 1}
    ).sort("computed_at", ASCENDING))

    dashboard = {
        "user_id":                user_id,
        "health_score":           health_record["health_score"] if health_record else None,
        "metrics":                health_record["metrics"]      if health_record else None,
        "latest_recommendations": latest_rec["recommendations"] if latest_rec    else None,
        "score_trend_30d": [
            {"date": r["computed_at"], "score": r["health_score"]["composite_score"]}
            for r in trend_records if "health_score" in r
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    return jsonify(dashboard), 200


@app.route("/users/gaps", methods=["GET"])
def get_users_with_gaps():
    gap_type       = request.args.get("gap_type")
    min_churn_risk = request.args.get("min_churn_risk", 50, type=float)

    pipeline = [
        {"$sort": {"computed_at": -1}},
        {"$group": {
            "_id":          "$user_id",
            "latest_score": {"$first": "$health_score"},
            "computed_at":  {"$first": "$computed_at"},
        }},
        {"$match": {"latest_score.churn_risk.score": {"$gte": min_churn_risk}}},
    ]
    if gap_type:
        pipeline[-1]["$match"]["latest_score.engagement_gaps"] = {"$in": [gap_type]}

    try:
        results = list(health_scores_col.aggregate(pipeline))
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    return jsonify({
        "users_with_gaps": [
            {
                "user_id":    r["_id"],
                "health_score": r["latest_score"]["composite_score"],
                "churn_risk": r["latest_score"]["churn_risk"],
                "gaps":       r["latest_score"]["engagement_gaps"],
                "computed_at": r["computed_at"],
            }
            for r in results
        ],
        "count": len(results),
    }), 200


@app.route("/users", methods=["GET"])
def list_users():
    """Return distinct user IDs that have health scores."""
    try:
        user_ids = health_scores_col.distinct("user_id")
    except PyMongoError as e:
        return jsonify({"error": "Database error", "detail": str(e)}), 500
    return jsonify({"users": user_ids, "count": len(user_ids)}), 200


def _mongo_ok() -> bool:
    try:
        mongo_client.admin.command("ping")
        return True
    except Exception:
        return False


@app.route("/health", methods=["GET"])
def health_check():
    mongo_up = _mongo_ok()
    return jsonify({
        "status":            "healthy" if mongo_up else "degraded",
        "mongo_connected":   mongo_up,
        "openai_configured": bool(OPENAI_API_KEY),
        "collections":       ["users", "events", "health_scores", "recommendations"],
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
