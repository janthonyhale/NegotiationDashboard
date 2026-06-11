import csv
import json
import re


ALLOWED = {'txt', 'json', 'jsonl', 'csv'}


def allowed(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_file(path, ext):
    def extract_irp_label(obj):
        if not isinstance(obj, dict):
            return None
        raw = obj.get('irp_label', obj.get('irp', obj.get('IRP')))
        if raw is None and isinstance(obj.get('meta'), dict):
            raw = obj['meta'].get('irp_label', obj['meta'].get('irp'))
        if raw is None:
            return None
        val = str(raw).strip()
        if not val:
            return None
        low = val.lower()
        if low.startswith('interest'):
            return 'Interest'
        if low.startswith('right'):
            return 'Right'
        if low.startswith('power'):
            return 'Power'
        return val[:1].upper() + val[1:]

    turns = []
    if ext == 'jsonl':
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                if not obj.get('speaker') and not obj.get('text'):
                    # Allow metadata/header rows in JSONL without treating them as dialogue turns.
                    continue
                irp_label = extract_irp_label(obj)
                meta = {'irp_label': irp_label} if irp_label else {}
                turns.append({'idx':len(turns),'speaker':obj.get('speaker','Unknown'),'text':obj.get('text',''),'ts':None,'meta':meta})
    elif ext == 'json':
        with open(path, encoding='utf-8-sig', errors='replace') as f: data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get('turns'), list):
            return data.get('turns', [])
        items = data if isinstance(data,list) else data.get('turns', data.get('messages',[data]))
        for i,obj in enumerate(items):
            irp_label = extract_irp_label(obj)
            meta = {'irp_label': irp_label} if irp_label else {}
            turns.append({'idx':i,'speaker':obj.get('speaker',obj.get('role','Unknown')),
                          'text':obj.get('text',obj.get('content','')),'ts':None,'meta':meta})
    elif ext == 'txt':
        with open(path, encoding='utf-8-sig', errors='replace') as f: content = f.read()
        pat = re.compile(r'^(Buyer|Seller|Mediator)(?:\s*[\[(]\s*(Interest|Power|Right)\s*[\])])?\s*:\s*(.+)', re.M|re.I)
        for i,m in enumerate(pat.finditer(content)):
            irp_label = m.group(2).capitalize() if m.group(2) else None
            meta = {'irp_label': irp_label} if irp_label else {}
            turns.append({'idx':i,'speaker':m.group(1).capitalize(),'text':m.group(3).strip(),'ts':None,'meta':meta})
        if not turns:
            for i,line in enumerate(content.strip().split('\n')):
                line=line.strip()
                if not line: continue
                turns.append({'idx':i,'speaker':'Buyer' if i%2==0 else 'Seller','text':line,'ts':None,'meta':{}})
    elif ext == 'csv':
        with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f)
            for i,row in enumerate(reader):
                speaker = row.get('speaker',row.get('Speaker',row.get('role','Unknown')))
                text    = row.get('text',row.get('Text',row.get('content','')))
                irp_label = extract_irp_label(row)
                meta = {'irp_label': irp_label} if irp_label else {}
                turns.append({'idx':i,'speaker':speaker,'text':text,'ts':None,'meta':meta})
    return turns


def extract_dialogue_language(path, ext, fallback='EN'):
    fallback = str(fallback or 'EN').upper()
    try:
        if ext == 'json':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                data = json.load(f)
            if isinstance(data, dict):
                raw = data.get('language', data.get('lang'))
                if isinstance(raw, str):
                    val = raw.strip().upper()
                    if val in {'EN', 'CN'}:
                        return val
        elif ext == 'jsonl':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                first = f.readline().strip()
            if first:
                obj = json.loads(first)
                if isinstance(obj, dict):
                    raw = obj.get('language', obj.get('lang'))
                    if isinstance(raw, str):
                        val = raw.strip().upper()
                        if val in {'EN', 'CN'}:
                            return val
        elif ext == 'csv':
            with open(path, newline='', encoding='utf-8-sig', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw = row.get('language', row.get('lang'))
                    if isinstance(raw, str):
                        val = raw.strip().upper()
                        if val in {'EN', 'CN'}:
                            return val
                    break
        elif ext == 'txt':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                first = (f.readline() or '').strip()
            m = re.match(r'^(language|lang)\s*[:=]\s*(EN|CN)\s*$', first, re.I)
            if m:
                return m.group(2).upper()
    except Exception:
        pass
    return fallback
