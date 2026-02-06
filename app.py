import os
import json
import csv
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_from_directory
)

# ----------------------------
# Flask setup
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


# ----------------------------
# Data model (simple)
# ----------------------------
# A "turn" looks like:
# {
#   "idx": 0,
#   "speaker": "Buyer" / "Seller" / "Mediator" / "Unknown",
#   "text": "...",
#   "ts": 1770359879.35963 (optional),
#   "meta": {...} (optional)
# }
#
# We'll store parsed turns in a server-side cache keyed by doc_id
DOC_CACHE: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# Heuristic placeholders (swap later)
# ----------------------------

EMOTION_LEXICON = {
    "anger": ["angry", "mad", "furious", "ridiculous", "unfair", "scam", "hate"],
    "sadness": ["sad", "upset", "disappointed", "hurt", "regret"],
    "joy": ["happy", "glad", "great", "awesome", "thanks", "thank you"],
    "fear": ["worried", "afraid", "concerned", "nervous", "anxious"],
}
NEGATIVE_WORDS = set(["unfair", "scam", "ridiculous", "never", "no way", "lawsuit", "fraud", "hate"])
THREAT_PATTERNS = [
    r"\blawsuit\b", r"\bsue\b", r"\blegal\b", r"\breport\b", r"\bchargeback\b", r"\bpolice\b"
]


def detect_language_and_country_placeholder(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Placeholder country/language detection.
    - If contains many CJK characters, label as 'zh' and country 'CN?'
    - Otherwise 'en' / 'Unknown'
    """
    text = " ".join([t.get("text", "") for t in turns])
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    total = max(1, len(text))
    cjk_ratio = cjk / total

    if cjk_ratio > 0.05:
        return {"language": "zh", "country_guess": "CN (heuristic)", "confidence": round(min(0.99, cjk_ratio * 8), 2)}
    return {"language": "en", "country_guess": "Unknown", "confidence": 0.3}


def emotion_scores_per_turn_placeholder(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Very simple lexicon-based emotion scoring per turn.
    Output: [{idx, anger, sadness, joy, fear, valence}]
    """
    out = []
    for t in turns:
        txt = (t.get("text") or "").lower()
        scores = {k: 0 for k in EMOTION_LEXICON.keys()}
        for emo, words in EMOTION_LEXICON.items():
            for w in words:
                if w in txt:
                    scores[emo] += 1

        # crude valence: joy - (anger+sadness+fear)
        valence = scores["joy"] - (scores["anger"] + scores["sadness"] + scores["fear"])
        out.append({"idx": t["idx"], **scores, "valence": valence})
    return out


def failure_risk_placeholder(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Placeholder "likelihood of failure".
    Heuristic: more negative words / threats -> higher risk.
    """
    joined = " ".join((t.get("text") or "").lower() for t in turns)
    neg_hits = sum(joined.count(w) for w in NEGATIVE_WORDS)
    threat_hits = sum(1 for pat in THREAT_PATTERNS if re.search(pat, joined))

    # Normalize into 0..1 in a hand-wavy way
    risk = min(1.0, 0.1 * neg_hits + 0.25 * threat_hits)
    label = "Low" if risk < 0.33 else ("Medium" if risk < 0.66 else "High")
    return {"risk": round(risk, 2), "label": label, "signals": {"neg_hits": neg_hits, "threat_hits": threat_hits}}


def advisor_interventions_placeholder(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mark turns where the system would advise intervention.
    Heuristic triggers:
      - Threat language
      - Strong negative valence
      - Rapid escalation: two consecutive negative-ish turns
    """
    emo = emotion_scores_per_turn_placeholder(turns)
    interventions = []
    prev_bad = False

    for t, e in zip(turns, emo):
        txt = (t.get("text") or "").lower()
        has_threat = any(re.search(pat, txt) for pat in THREAT_PATTERNS)
        negative = (e["valence"] <= -1) or any(w in txt for w in NEGATIVE_WORDS)

        reason = []
        if has_threat:
            reason.append("Threat/legal language")
        if e["valence"] <= -1:
            reason.append("Negative emotion spike")
        if prev_bad and negative:
            reason.append("Escalation (consecutive negative turns)")

        should = bool(reason)
        interventions.append({
            "idx": t["idx"],
            "should_intervene": should,
            "reason": reason,
            "suggestion": "Placeholder: advise de-escalation / reframe / ask clarifying question."
                         if should else ""
        })
        prev_bad = negative

    return interventions


def translate_placeholder(text: str, target_lang: str = "en") -> Dict[str, Any]:
    """
    Placeholder translation stub.
    """
    return {
        "target_lang": target_lang,
        "translated_text": f"[TRANSLATION PLACEHOLDER → {target_lang}] {text}"
    }


# ----------------------------
# Parsing uploads
# ----------------------------

def safe_read_text(path: Path, max_chars: int = 2_000_000) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return raw[:max_chars]


def parse_jsonl_turns(text: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        # Try common keys; fallback to best-effort
        speaker = obj.get("role") or obj.get("speaker") or obj.get("agent") or obj.get("type") or "Unknown"
        msg = obj.get("text") or obj.get("message") or obj.get("content") or ""
        ts = obj.get("ts") or obj.get("timestamp")

        if isinstance(speaker, str):
            # normalize a bit
            if speaker.lower() in ["buyer", "seller", "mediator"]:
                speaker = speaker.capitalize()

        turns.append({
            "idx": len(turns),
            "speaker": speaker if speaker else "Unknown",
            "text": str(msg),
            "ts": ts,
            "meta": {k: v for k, v in obj.items() if k not in ["role", "speaker", "agent", "type", "text", "message", "content", "ts", "timestamp"]}
        })
    return turns


def parse_txt_turns(text: str) -> List[Dict[str, Any]]:
    """
    Parse formats like:
      Buyer: hello
      Seller: hi
    or
      [Buyer] hello
    """
    turns: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m1 = re.match(r"^(Buyer|Seller|Mediator)\s*:\s*(.*)$", line, flags=re.I)
        m2 = re.match(r"^\[(Buyer|Seller|Mediator)\]\s*(.*)$", line, flags=re.I)

        if m1:
            speaker = m1.group(1).capitalize()
            msg = m1.group(2)
        elif m2:
            speaker = m2.group(1).capitalize()
            msg = m2.group(2)
        else:
            speaker = "Unknown"
            msg = line

        turns.append({"idx": len(turns), "speaker": speaker, "text": msg, "ts": None, "meta": {}})
    return turns


def parse_csv_turns(path: Path) -> List[Dict[str, Any]]:
    """
    Expect columns like:
      speaker,text,ts
    but tolerate variations: role/content/message
    """
    turns: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker = row.get("speaker") or row.get("role") or row.get("agent") or "Unknown"
            msg = row.get("text") or row.get("message") or row.get("content") or ""
            ts = row.get("ts") or row.get("timestamp") or None

            if isinstance(speaker, str) and speaker.lower() in ["buyer", "seller", "mediator"]:
                speaker = speaker.capitalize()

            turns.append({"idx": len(turns), "speaker": speaker, "text": msg, "ts": ts, "meta": row})
    return turns


def parse_upload(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return parse_jsonl_turns(safe_read_text(path))
    if ext == ".json":
        # If it's a list of turns, parse it. Else attempt fallback.
        raw = safe_read_text(path)
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                turns = []
                for item in obj:
                    if not isinstance(item, dict):
                        continue
                    speaker = item.get("role") or item.get("speaker") or "Unknown"
                    msg = item.get("text") or item.get("message") or item.get("content") or ""
                    ts = item.get("ts") or item.get("timestamp")
                    if isinstance(speaker, str) and speaker.lower() in ["buyer", "seller", "mediator"]:
                        speaker = speaker.capitalize()
                    turns.append({"idx": len(turns), "speaker": speaker, "text": str(msg), "ts": ts, "meta": item})
                return turns
        except Exception:
            pass
        # fallback: treat as jsonl-ish
        return parse_jsonl_turns(raw)
    if ext == ".csv":
        return parse_csv_turns(path)
    # default treat as text
    return parse_txt_turns(safe_read_text(path))


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def index():
    return redirect(url_for("upload_page"))


@app.get("/upload")
def upload_page():
    return render_template("upload.html")


@app.post("/upload")
def upload_post():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Please choose a file to upload.", "error")
        return redirect(url_for("upload_page"))

    ext = Path(f.filename).suffix.lower()
    if ext not in [".jsonl", ".json", ".txt", ".csv"]:
        flash("Supported file types: .jsonl, .json, .txt, .csv", "error")
        return redirect(url_for("upload_page"))

    doc_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{doc_id}{ext}"
    f.save(saved_path)

    turns = parse_upload(saved_path)
    if not turns:
        flash("Could not parse any turns from that file.", "error")
        return redirect(url_for("upload_page"))

    # Compute placeholders
    lang_country = detect_language_and_country_placeholder(turns)
    emo = emotion_scores_per_turn_placeholder(turns)
    risk = failure_risk_placeholder(turns)
    interventions = advisor_interventions_placeholder(turns)

    DOC_CACHE[doc_id] = {
        "doc_id": doc_id,
        "filename": f.filename,
        "saved_path": str(saved_path),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "turns": turns,
        "lang_country": lang_country,
        "emotion": emo,
        "risk": risk,
        "interventions": interventions,
    }

    session["doc_id"] = doc_id
    return redirect(url_for("dashboard", doc_id=doc_id))


@app.get("/dashboard/<doc_id>")
def dashboard(doc_id: str):
    doc = DOC_CACHE.get(doc_id)
    if not doc:
        flash("Document not found (server cache reset?). Re-upload.", "error")
        return redirect(url_for("upload_page"))
    return render_template("dashboard.html", doc=doc)


@app.get("/api/doc/<doc_id>")
def api_doc(doc_id: str):
    doc = DOC_CACHE.get(doc_id)
    if not doc:
        return jsonify({"error": "not_found"}), 404

    # Return minimal payload needed by frontend
    return jsonify({
        "doc_id": doc_id,
        "filename": doc["filename"],
        "turns": doc["turns"],
        "lang_country": doc["lang_country"],
        "emotion": doc["emotion"],
        "risk": doc["risk"],
        "interventions": doc["interventions"],
    })


@app.post("/api/translate")
def api_translate():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    target = data.get("target_lang", "en")
    return jsonify(translate_placeholder(text, target_lang=target))


@app.get("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    # For local dev:
    #   python app.py
    # Then open http://127.0.0.1:5000/upload
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)
