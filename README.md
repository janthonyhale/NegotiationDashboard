# Negotiation Dashboard (Local Flask)

## Setup
```bash
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
python app.py
```
Then open: http://127.0.0.1:5000/upload

## Features
- Upload transcript (.jsonl/.json/.txt/.csv)
- Pre-summary page with Pareto frontier + preferences
- Step-through negotiation (no future utterances revealed)
- Emotion line chart (Buyer+Seller) with emotion toggle
- Country-of-origin prediction panel with a simple highlighted map
- Post-summary page with Pareto + final solution point
- Download PDF summary (includes all figures)
