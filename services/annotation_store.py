import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.getenv('ANNOTATION_CORRECTIONS_DB', os.path.join(os.getcwd(), 'annotation_corrections.sqlite3'))


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_annotation_db():
    os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)
    with _conn() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS annotation_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                turn_idx INTEGER NOT NULL,
                speaker TEXT,
                utterance TEXT,
                annotation_type TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            '''
        )
        conn.execute('CREATE INDEX IF NOT EXISTS idx_annotation_type ON annotation_corrections(annotation_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_filename_turn ON annotation_corrections(filename, turn_idx)')


def record_annotation_correction(filename, turn_idx, speaker, utterance, annotation_type, old_value, new_value):
    init_annotation_db()
    old_json = json.dumps(old_value, ensure_ascii=False, sort_keys=True) if not isinstance(old_value, str) else old_value
    new_json = json.dumps(new_value, ensure_ascii=False, sort_keys=True) if not isinstance(new_value, str) else new_value
    created_at = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            '''
            INSERT INTO annotation_corrections
              (filename, turn_idx, speaker, utterance, annotation_type, old_value, new_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (filename, int(turn_idx), speaker, utterance, annotation_type, old_json, new_json, created_at),
        )
        return {'id': cur.lastrowid, 'created_at': created_at}


def recent_annotation_corrections(annotation_type=None, limit=12):
    init_annotation_db()
    sql = 'SELECT * FROM annotation_corrections'
    params = []
    if annotation_type:
        sql += ' WHERE annotation_type = ?'
        params.append(annotation_type)
    sql += ' ORDER BY id DESC LIMIT ?'
    params.append(int(limit))
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        for key in ['old_value', 'new_value']:
            try:
                item[key] = json.loads(item[key])
            except Exception:
                pass
        out.append(item)
    return out


def format_corrections_for_prompt(annotation_type=None, limit=8):
    examples = recent_annotation_corrections(annotation_type=annotation_type, limit=limit)
    if not examples:
        return 'No prior human annotation corrections are available yet.'
    lines = []
    for ex in examples:
        utterance = str(ex.get('utterance') or '')[:240]
        lines.append(
            f"- {ex.get('annotation_type')} correction: speaker={ex.get('speaker')}, "
            f"utterance={utterance!r}, old={ex.get('old_value')}, human_corrected={ex.get('new_value')}"
        )
    return '\n'.join(lines)
