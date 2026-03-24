import os, json, csv, io, re, base64
import urllib.request
import urllib.error
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch

from predictor import RegionPredictor

app = Flask(__name__)
app.secret_key = 'nego_dash_kodis_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED = {'txt', 'json', 'jsonl', 'csv'}

# ── KODIS Issue Definitions ───────────────────────────────────────────────────
KODIS_ISSUES = {
    'refund':          {'label': 'Refund',              'options': ['Full (100%)', 'Half (50%)', 'None (0%)'],  'values': [1.0, 0.5, 0.0]},
    'buyer_review':    {'label': 'Buyer Review Removed', 'options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
    'seller_review':   {'label': 'Seller Review Removed','options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
    'seller_apology':  {'label': 'Receive Apology',    'options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
    'buyer_apology':   {'label': 'Buyer Apologizes',    'options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
}

DEFAULT_BUYER_WEIGHTS  = {'refund': 50, 'buyer_review': 10, 'seller_review': 25, 'seller_apology': 15, 'buyer_apology': 0}
DEFAULT_SELLER_WEIGHTS = {'refund': 40, 'buyer_review': 30, 'seller_review': 5,  'seller_apology': 0,  'buyer_apology': 25}

def kodis_utility(outcome, weights, role):
    """
    outcome: dict with keys refund, buyer_review, seller_review, seller_apology, buyer_apology
    weights: dict with preference weights 0-100 per issue
    role: 'buyer' or 'seller'
    """
    total_w = sum(weights.values()) or 1
    if role == 'buyer':
        # Buyer wants: high refund, seller review removed, seller apology, buyer review NOT removed
        u  = weights['refund']         * outcome['refund']
        u += weights['seller_review']  * outcome['seller_review']
        u += weights['seller_apology'] * outcome['seller_apology']
        u += weights['buyer_review']   * (1 - outcome['buyer_review'])  # buyer doesn't want own review removed
        u += weights['buyer_apology']  * (1 - outcome['buyer_apology']) # buyer doesn't want to apologize
    else:  # seller
        # Seller wants: low refund, buyer review removed, buyer apology, seller review NOT removed
        u  = weights['refund']         * (1 - outcome['refund'])
        u += weights['buyer_review']   * outcome['buyer_review']
        u += weights['buyer_apology']  * outcome['buyer_apology']
        u += weights['seller_review']  * (1 - outcome['seller_review'])  # seller doesn't want own review removed
        u += weights['seller_apology'] * (1 - outcome['seller_apology']) # seller doesn't want to apologize
    return round(u / total_w * 100, 1)

def generate_all_outcomes(bw, sw):
    """Generate all 3×2×2×2×2=48 outcomes with buyer/seller utility."""
    outcomes = []
    for rval in [1.0, 0.5, 0.0]:
        for brr in [1, 0]:
            for srr in [1, 0]:
                for sa in [1, 0]:
                    for ba in [1, 0]:
                        outcome = {'refund': rval, 'buyer_review': brr, 'seller_review': srr,
                                   'seller_apology': sa, 'buyer_apology': ba}
                        bu = kodis_utility(outcome, bw, 'buyer')
                        su = kodis_utility(outcome, sw, 'seller')
                        outcomes.append({**outcome, 'buyer_util': bu, 'seller_util': su,
                                         'refund_label': {1.0:'Full',0.5:'Half',0.0:'None'}[rval]})
    return outcomes

def compute_pareto(outcomes):
    pareto = []
    for s in outcomes:
        dominated = any(
            o['buyer_util'] >= s['buyer_util'] and o['seller_util'] >= s['seller_util'] and
            (o['buyer_util'] > s['buyer_util'] or o['seller_util'] > s['seller_util'])
            for o in outcomes if o is not s
        )
        if not dominated:
            pareto.append(s)
    return sorted(pareto, key=lambda x: x['buyer_util'])

# ── Emotion Keywords ──────────────────────────────────────────────────────────
EMO_KEYS = {
    'anger':      ['angry','furious','frustrated','unacceptable','ridiculous','demand','insist','refuse','absurd','outrageous','never','scam','lied','terrible','worst'],
    'fear':       ['worried','concerned','anxious','risk','uncertain','doubt','afraid','nervous','unsure','hesitant','danger','scared','apprehensive'],
    'sadness':    ['sad','disappointed','upset','heartbroken','regret','sorry to hear','depressed','unhappy','let down','hurt'],
    'joy':        ['great','excellent','happy','pleased','wonderful','perfect','agree','deal','good','fantastic','love','glad','thrilled','excited','satisfied','appreciate','thank'],
    'surprise':   ['wow','unexpected','surprising','shocked','amazed','incredible','unbelievable','suddenly','wait','actually','really','seriously'],
    'compassion': ['understand','sorry','apologize','empathize','appreciate your','i see','i hear','difficult','hard for you','must be','feel for'],
    'neutral':    ['the','is','are','and','but','however','therefore','perhaps','maybe','could','would','should'],
}
NEG_SIGNALS = ['walkaway','walk away','final','no deal','reject','refuse','unacceptable','sue','lawyer','court','legal action','worst','scam']

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


def llm_translate_cn_to_en(text):
    api_key = os.getenv('OPENAI_API_KEY')
    if not text:
        return ''
    if not api_key:
        return '[Translation unavailable: set OPENAI_API_KEY]'
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "Translate Simplified or Traditional Chinese into natural concise English. Return only the translation text."},
            {"role": "user", "content": text}
        ],
        "temperature": 0
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return '[Translation unavailable]'


CN_PROVINCES = {
    'beijing': ['beijing', '北京'], 'tianjin': ['tianjin', '天津'], 'hebei': ['hebei', '河北'],
    'shanxi': ['shanxi', '山西'], 'inner mongolia': ['inner mongolia', 'neimenggu', '内蒙古'],
    'liaoning': ['liaoning', '辽宁'], 'jilin': ['jilin', '吉林'], 'heilongjiang': ['heilongjiang', '黑龙江'],
    'shanghai': ['shanghai', '上海'], 'jiangsu': ['jiangsu', '江苏'], 'zhejiang': ['zhejiang', '浙江'],
    'anhui': ['anhui', '安徽'], 'fujian': ['fujian', '福建'], 'jiangxi': ['jiangxi', '江西'],
    'shandong': ['shandong', '山东'], 'henan': ['henan', '河南'], 'hubei': ['hubei', '湖北'],
    'hunan': ['hunan', '湖南'], 'guangdong': ['guangdong', '广东'], 'guangxi': ['guangxi', '广西'],
    'hainan': ['hainan', '海南'], 'chongqing': ['chongqing', '重庆'], 'sichuan': ['sichuan', '四川'],
    'guizhou': ['guizhou', '贵州'], 'yunnan': ['yunnan', '云南'], 'tibet': ['tibet', 'xizang', '西藏'],
    'shaanxi': ['shaanxi', '陕西'], 'gansu': ['gansu', '甘肃'], 'qinghai': ['qinghai', '青海'],
    'ningxia': ['ningxia', '宁夏'], 'xinjiang': ['xinjiang', '新疆'], 'hong kong': ['hong kong', '香港'],
    'macau': ['macau', '澳门'], 'taiwan': ['taiwan', '台湾']
}

CN_PROVINCE_CENTROIDS = {
    'beijing': (116.40, 39.90), 'tianjin': (117.20, 39.13), 'hebei': (114.53, 38.04),
    'shanxi': (112.55, 37.87), 'inner mongolia': (111.67, 40.82), 'liaoning': (123.43, 41.80),
    'jilin': (125.32, 43.90), 'heilongjiang': (126.53, 45.80), 'shanghai': (121.47, 31.23),
    'jiangsu': (118.78, 32.07), 'zhejiang': (120.15, 30.28), 'anhui': (117.27, 31.86),
    'fujian': (119.30, 26.08), 'jiangxi': (115.89, 28.68), 'shandong': (117.00, 36.65),
    'henan': (113.63, 34.75), 'hubei': (114.31, 30.60), 'hunan': (112.98, 28.20),
    'guangdong': (113.27, 23.13), 'guangxi': (108.32, 22.82), 'hainan': (110.35, 20.02),
    'chongqing': (106.55, 29.56), 'sichuan': (104.07, 30.67), 'guizhou': (106.71, 26.58),
    'yunnan': (102.71, 25.04), 'tibet': (91.11, 29.97), 'shaanxi': (108.95, 34.27),
    'gansu': (103.84, 36.06), 'qinghai': (101.78, 36.62), 'ningxia': (106.23, 38.48),
    'xinjiang': (87.62, 43.82), 'hong kong': (114.17, 22.32), 'macau': (113.54, 22.20),
    'taiwan': (121.56, 25.04),
}

CHINA_GEOJSON_CANDIDATES = [
    os.path.join(os.getcwd(), 'china_provinces.geojson'),
    os.path.join(os.getcwd(), 'data', 'china_provinces.geojson'),
    os.path.join(os.getcwd(), 'uploads', 'china_provinces.geojson'),
]
_CHINA_GEOJSON_CACHE = None

CN_PROVINCE_ALIASES = {
    'beijing': 'beijing', '北京市': 'beijing',
    'tianjin': 'tianjin', '天津市': 'tianjin',
    'hebei': 'hebei', '河北省': 'hebei',
    'shanxi': 'shanxi', '山西省': 'shanxi',
    'inner mongolia': 'inner mongolia', 'neimenggu': 'inner mongolia', '内蒙古自治区': 'inner mongolia', '内蒙古': 'inner mongolia',
    'liaoning': 'liaoning', '辽宁省': 'liaoning',
    'jilin': 'jilin', '吉林省': 'jilin',
    'heilongjiang': 'heilongjiang', '黑龙江省': 'heilongjiang',
    'shanghai': 'shanghai', '上海市': 'shanghai',
    'jiangsu': 'jiangsu', '江苏省': 'jiangsu',
    'zhejiang': 'zhejiang', '浙江省': 'zhejiang',
    'anhui': 'anhui', '安徽省': 'anhui',
    'fujian': 'fujian', '福建省': 'fujian',
    'jiangxi': 'jiangxi', '江西省': 'jiangxi',
    'shandong': 'shandong', '山东省': 'shandong',
    'henan': 'henan', '河南省': 'henan',
    'hubei': 'hubei', '湖北省': 'hubei',
    'hunan': 'hunan', '湖南省': 'hunan',
    'guangdong': 'guangdong', '广东省': 'guangdong',
    'guangxi': 'guangxi', '广西壮族自治区': 'guangxi', '广西': 'guangxi',
    'hainan': 'hainan', '海南省': 'hainan',
    'chongqing': 'chongqing', '重庆市': 'chongqing',
    'sichuan': 'sichuan', '四川省': 'sichuan',
    'guizhou': 'guizhou', '贵州省': 'guizhou',
    'yunnan': 'yunnan', '云南省': 'yunnan',
    'tibet': 'tibet', '西藏自治区': 'tibet', 'xizang': 'tibet', '西藏': 'tibet',
    'shaanxi': 'shaanxi', '陕西省': 'shaanxi',
    'gansu': 'gansu', '甘肃省': 'gansu',
    'qinghai': 'qinghai', '青海省': 'qinghai',
    'ningxia': 'ningxia', '宁夏回族自治区': 'ningxia', '宁夏': 'ningxia',
    'xinjiang': 'xinjiang', '新疆维吾尔自治区': 'xinjiang', '新疆': 'xinjiang',
    'hong kong': 'hong kong', '香港特别行政区': 'hong kong', '香港': 'hong kong',
    'macau': 'macau', '澳门特别行政区': 'macau', '澳门': 'macau',
    'taiwan': 'taiwan', '台湾省': 'taiwan', '台湾': 'taiwan',
}


def _normalize_cn_province_name(raw):
    key = str(raw or '').strip()
    if not key:
        return None
    return CN_PROVINCE_ALIASES.get(key.lower(), CN_PROVINCE_ALIASES.get(key, None))


def _extract_name_from_feature(feature):
    props = feature.get('properties', {}) if isinstance(feature, dict) else {}
    for key in ['name', 'NAME_1', 'NL_NAME_1', 'province', 'prov_name', 'NAME_CHN', 'name_zh', 'NAME']:
        if key in props:
            normalized = _normalize_cn_province_name(props.get(key))
            if normalized:
                return normalized
    return None


def _extract_polygon_rings(geometry):
    if not isinstance(geometry, dict):
        return []
    gtype = geometry.get('type')
    coords = geometry.get('coordinates', [])
    rings = []
    if gtype == 'Polygon':
        if coords:
            rings.append(coords[0])
    elif gtype == 'MultiPolygon':
        for poly in coords:
            if poly and poly[0]:
                rings.append(poly[0])
    return rings


def _load_china_geojson_features():
    global _CHINA_GEOJSON_CACHE
    if _CHINA_GEOJSON_CACHE is not None:
        return _CHINA_GEOJSON_CACHE
    env_path = os.getenv('CHINA_GEOJSON_PATH')
    candidates = [env_path] if env_path else []
    candidates.extend(CHINA_GEOJSON_CANDIDATES)
    for path in candidates:
        if not path:
            continue
        try:
            with open(path, encoding='utf-8') as f:
                gj = json.load(f)
            feats = gj.get('features', []) if isinstance(gj, dict) else []
            if isinstance(feats, list) and feats:
                _CHINA_GEOJSON_CACHE = feats
                return feats
        except Exception:
            continue
    _CHINA_GEOJSON_CACHE = []
    return _CHINA_GEOJSON_CACHE

def infer_cn_province_distribution(turns, role):
    role_turns = [str(t.get('text', '')) for t in turns if str(t.get('speaker', '')).lower() == role.lower()]
    text = ' '.join(role_turns).lower()

    # Start with a light, uniform prior so every province can be visualized.
    counts = {province: 0.2 for province in CN_PROVINCES.keys()}
    for province, aliases in CN_PROVINCES.items():
        hit = sum(text.count(alias.lower()) for alias in aliases)
        if hit > 0:
            counts[province] += float(hit) * 3.0

    # Nudge major-language hubs slightly when no clear location cues exist.
    if not any(v > 0.2 for v in counts.values()):
        for hub in ['guangdong', 'beijing', 'shanghai', 'sichuan', 'zhejiang']:
            counts[hub] += 0.8

    total = sum(counts.values()) or 1.0
    return {k: round(v / total, 6) for k, v in counts.items()}


CN_REGION_PROVINCES = {
    # Keep province scope aligned with chinese model classes in predictor.py.
    'North': ['beijing', 'tianjin', 'hebei', 'shandong'],
    'Central': ['henan', 'hubei', 'hunan', 'sichuan'],
    'Wu_Min': ['shanghai', 'jiangsu', 'zhejiang', 'fujian'],
    'Xian_Yue': ['guangdong', 'guangxi', 'hainan', 'hong kong'],
}

CN_REGION_LABEL_POS = {
    'North': (116.0, 39.2),
    'Central': (112.8, 31.5),
    'Wu_Min': (120.2, 30.2),
    'Xian_Yue': (113.2, 23.0),
}


def project_region_probs_to_provinces(region_probs, base_province_probs):
    region_map = CN_REGION_PROVINCES
    out = {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()}
    base = {k: float(v) for k, v in (base_province_probs or {}).items() if k in out}
    for region, provinces in region_map.items():
        rprob = float((region_probs or {}).get(region, 0.0))
        if rprob <= 0:
            continue
        weights = {p: max(0.0, base.get(p, 0.0)) for p in provinces}
        wsum = sum(weights.values())
        if wsum <= 0:
            even = rprob / max(len(provinces), 1)
            for p in provinces:
                out[p] += even
        else:
            for p, w in weights.items():
                out[p] += rprob * (w / wsum)
    total = sum(out.values()) or 1.0
    return {k: round(v / total, 6) for k, v in out.items()}


def constrain_cn_probs_to_model_scope(province_probs):
    scoped = {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()}
    allowed = {p for plist in CN_REGION_PROVINCES.values() for p in plist}
    for k in allowed:
        scoped[k] = float((province_probs or {}).get(k, 0.0))
    total = sum(scoped.values())
    if total <= 0:
        even = 1.0 / len(allowed)
        for k in allowed:
            scoped[k] = round(even, 6)
        return scoped
    return {k: round(v / total, 6) for k, v in scoped.items()}


def make_cn_province_map(province_probs, role='buyer', region_probs=None):
    probs = {k: float(v) for k, v in (province_probs or {}).items() if k in CN_PROVINCE_CENTROIDS}
    if not probs:
        return None
    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}

    bg = '#07111f'
    primary = '#4f91ff' if str(role).lower() == 'buyer' else '#f43f5e'

    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=bg)
    ax.set_facecolor(bg)
    features = _load_china_geojson_features()
    drew_geojson = False
    min_lon, max_lon, min_lat, max_lat = 200, -200, 90, -90

    if str(role).lower() == 'buyer':
        palette = ['#10253d', '#1f4f8c', '#3b82f6', '#93c5fd']
    else:
        palette = ['#31131b', '#7f1d2d', '#f43f5e', '#fda4af']

    rp = dict(region_probs or {})
    if not rp:
        rp = {}
        for region, plist in CN_REGION_PROVINCES.items():
            rp[region] = float(sum(probs.get(p, 0.0) for p in plist))
        total_r = sum(rp.values()) or 1.0
        rp = {k: (v / total_r) for k, v in rp.items()}

    province_to_region = {}
    for region, plist in CN_REGION_PROVINCES.items():
        for p in plist:
            province_to_region[p] = region

    def color_for_prob(p):
        p = max(0.0, min(1.0, p))
        if p <= 0:
            return '#6b7280'
        if p >= 0.20:
            return palette[3]
        if p >= 0.08:
            return palette[2]
        if p >= 0.03:
            return palette[1]
        return palette[0]

    if features:
        for feat in features:
            pname = _extract_name_from_feature(feat)
            rings = _extract_polygon_rings(feat.get('geometry', {}))
            if not rings:
                continue
            region_name = province_to_region.get(pname)
            p = float(rp.get(region_name, 0.0)) if region_name else 0.0
            face = color_for_prob(p)
            for ring in rings:
                if not ring:
                    continue
                xs = [pt[0] for pt in ring if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                ys = [pt[1] for pt in ring if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                if not xs or not ys:
                    continue
                min_lon, max_lon = min(min_lon, min(xs)), max(max_lon, max(xs))
                min_lat, max_lat = min(min_lat, min(ys)), max(max_lat, max(ys))
                poly = mpatches.Polygon(list(zip(xs, ys)), closed=True, facecolor=face, edgecolor='#3d5f83', linewidth=0.45, alpha=0.92)
                ax.add_patch(poly)
                drew_geojson = True

        for region, (lon, lat) in CN_REGION_LABEL_POS.items():
            p = float(rp.get(region, 0.0))
            if p <= 0:
                continue
            ax.text(
                lon + 0.15, lat + 0.10, f"{region} {p * 100:.1f}%",
                color='#e6f0ff', fontsize=13.0, fontweight='bold', zorder=6
            )

    if not drew_geojson:
        # Fallback if geojson is missing or malformed.
        lons = []
        lats = []
        sizes = []
        colors = []
        for prov, (lon, lat) in CN_PROVINCE_CENTROIDS.items():
            region_name = province_to_region.get(prov)
            p = float(rp.get(region_name, 0.0)) if region_name else 0.0
            lons.append(lon)
            lats.append(lat)
            sizes.append(140 + p * 2800)
            colors.append(primary if p > 0.001 else '#27425e')

        ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.88, edgecolors='#dbeafe', linewidths=0.35, zorder=3)
        min_lon, max_lon, min_lat, max_lat = 72, 136, 17, 54

    ax.set_xlim(min_lon - 2.0, max_lon + 2.0)
    ax.set_ylim(min_lat - 1.5, max_lat + 1.5)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axis_off()
    fig.tight_layout()
    return fig_b64(fig)

def score_emo(text):
    tl = text.lower()
    s = {}
    for emo, words in EMO_KEYS.items():
        if emo == 'neutral':
            count = sum(1 for w in words if f' {w} ' in f' {tl} ')
            s[emo] = max(0.0, min(0.3 + count * 0.02, 1.0))
        else:
            count = sum(1 for w in words if w in tl)
            s[emo] = min(count * 0.28, 1.0)
    total = sum(s.values()) or 1
    # Normalize so they sum to ~1 but cap each
    for k in s: s[k] = round(s[k] / total * min(total, 1.5), 3)
    # Add valence
    neg = s['anger'] + s['fear'] + s.get('sadness', 0)
    pos = s['joy'] + s['compassion']
    s['valence'] = round(max(-1, min(1, pos - neg)), 3)
    return s

def enrich(turns):
    for t in turns:
        tl = t['text'].lower()
        t['emotions'] = score_emo(t['text'])
        t['negative_signals'] = sum(1 for s in NEG_SIGNALS if s in tl)
        t['threat_signals']   = sum(1 for s in ['sue','lawyer','court','legal action'] if s in tl)
    return turns





def _default_pre_dispute_justifications():
    return {
        'refund': {'buyer': '', 'seller': ''},
        'buyer_review': {'buyer': '', 'seller': ''},
        'seller_review': {'buyer': '', 'seller': ''},
        'receive_apology': {'buyer': '', 'seller': ''},
    }


def _normalize_justification_payload(raw):
    out = _default_pre_dispute_justifications()
    if not isinstance(raw, dict):
        return out
    aliases = {
        'refund': 'refund',
        'buyer_review': 'buyer_review',
        'seller_review': 'seller_review',
        'seller_apology': 'receive_apology',
        'buyer_apology': 'receive_apology',
        'receive_apology': 'receive_apology',
        'apology': 'receive_apology',
    }
    for k, v in raw.items():
        key = aliases.get(str(k).strip(), str(k).strip())
        if key not in out or not isinstance(v, dict):
            continue
        b = v.get('buyer')
        s = v.get('seller')
        if isinstance(b, str) and b.strip():
            out[key]['buyer'] = b.strip()
        if isinstance(s, str) and s.strip():
            out[key]['seller'] = s.strip()
    return out


def extract_pre_dispute_justifications(path, ext):
    defaults = _default_pre_dispute_justifications()
    try:
        if ext == 'json':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ['pre_dispute_justifications', 'justifications', 'issue_justifications']:
                    if key in data:
                        return _normalize_justification_payload(data.get(key))
        elif ext == 'jsonl':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                first = f.readline().strip()
            if first:
                obj = json.loads(first)
                if isinstance(obj, dict) and any(k in obj for k in ['pre_dispute_justifications', 'justifications', 'issue_justifications']):
                    for key in ['pre_dispute_justifications', 'justifications', 'issue_justifications']:
                        if key in obj:
                            return _normalize_justification_payload(obj.get(key))
    except Exception:
        pass
    return defaults


def _normalize_weight_payload(raw, defaults):
    out = dict(defaults)
    if not isinstance(raw, dict):
        return out
    aliases = {
        'refund': 'refund',
        'buyer_review': 'buyer_review',
        'seller_review': 'seller_review',
        'seller_apology': 'seller_apology',
        'receive_apology': 'seller_apology',
        'buyer_apology': 'buyer_apology',
    }
    valid_keys = {'refund', 'buyer_review', 'seller_review', 'seller_apology', 'buyer_apology'}
    for k, v in raw.items():
        key = aliases.get(str(k).strip(), str(k).strip())
        if key not in out and key not in valid_keys:
            continue
        try:
            out[key] = max(0, min(100, int(float(v))))
        except Exception:
            continue
    return out


def extract_preference_weights(path, ext):
    bw = {}
    sw = {}
    try:
        if ext == 'json':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                data = json.load(f)
        elif ext == 'jsonl':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                first = f.readline().strip()
            data = json.loads(first) if first else {}
        else:
            data = {}
        if isinstance(data, dict):
            raw_b = data.get('buyer_weights', data.get('buyer_preferences'))
            raw_s = data.get('seller_weights', data.get('seller_preferences'))
            if isinstance(raw_b, dict):
                bw = _normalize_weight_payload(raw_b, {})
            if isinstance(raw_s, dict):
                sw = _normalize_weight_payload(raw_s, {})
    except Exception:
        pass
    return bw, sw



def turns_have_cached_enrichment(turns):
    if not isinstance(turns, list) or not turns:
        return False
    for t in turns:
        if not isinstance(t, dict):
            continue
        if t.get('emotions') and 'negative_signals' in t and 'threat_signals' in t:
            return True
    return False

def llm_emotion_scores(text):
    """Use GPT-4o emotion classification when OPENAI_API_KEY is available.
    Falls back to keyword heuristic on error/missing key.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return score_emo(text)

    system_prompt = """You are a good emotion classification tool. Your task is to classify the emotion of the last speaker based on the contextual dialogue.
Your output should be a JSON object with an 'emotion' field, categorizing the dialogue with a score for each: joy, anger, fear, sadness, surprise, compassion, or neutral.
These scores should sum to one. If an utterance is neutral, then neutral must be one with every other label set to zero."""

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"This is the context: {text}. What is the emotion of the current speaker?"}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type':'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        content = data['choices'][0]['message']['content']
        obj = json.loads(content)
        emo = obj.get('emotion', obj)
        out = {
            'joy': float(emo.get('joy',0)),
            'anger': float(emo.get('anger',0)),
            'fear': float(emo.get('fear',0)),
            'sadness': float(emo.get('sadness',0)),
            'surprise': float(emo.get('surprise',0)),
            'compassion': float(emo.get('compassion',0)),
            'neutral': float(emo.get('neutral',0)),
        }
        for k in out:
            out[k] = max(0.0, min(1.0, out[k]))

        total = sum(out.values())
        if total <= 0:
            return score_emo(text)
        out = {k: round(v / total, 3) for k, v in out.items()}
        if out['neutral'] >= 0.999:
            out = {k: 0.0 for k in out}
            out['neutral'] = 1.0

        neg = out['anger'] + out['fear'] + out['sadness']
        pos = out['joy'] + out['compassion']
        out['valence'] = round(max(-1, min(1, pos - neg)), 3)
        return out
    except Exception:
        return score_emo(text)


def llm_operational_summary(turns_so_far, country_snapshot, risk_snapshot):
    """Generate a succinct IRP-aware operational summary for the current state."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not turns_so_far:
        return 'No turns processed yet. Step through the dialogue to generate an operational summary.'

    recent_turns = turns_so_far[-40:]
    transcript_excerpt = "\n".join(
        f"{t.get('speaker','Unknown')}: {t.get('text','')} | emo={t.get('emotions',{})}"
        for t in recent_turns
    )

    if not api_key:
        return (
            f"Risk is {risk_snapshot.get('label','Unknown')} ({risk_snapshot.get('score',0)}), with the dialogue still mixing rights and power claims over core interests. "
            "Emotions remain active, so focus next on de-escalation and one concrete reciprocal package tied to shared interests."
        )

    prompt = (
        "Write an operational summary in at most TWO sentences. "
        "Only highlight the most important facets of the dialogue so far. "
        "Use signals from: integrative potential, dialogue history progression, emotional trajectory, and IRP balance (Interests/Rights/Power). "
        "Keep it high-level, practical, and easy to understand for mediators. "
        "Treat country predictions as uncertain priors (not facts).\n\n"
        f"Risk snapshot: {json.dumps(risk_snapshot)}\n"
        f"Country snapshot: {json.dumps(country_snapshot)}\n"
        f"Observed turns:\n{transcript_excerpt}"
    )

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You summarize negotiation states for mediators in plain, crisp language."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 110
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return (
            f"Risk is {risk_snapshot.get('label','Unknown')} ({risk_snapshot.get('score',0)}), with the dialogue still mixing rights and power claims over core interests. "
            "Emotions remain active, so focus next on de-escalation and one concrete reciprocal package tied to shared interests."
        )


def llm_evolution_summary(op_summaries):
    """Summarize how operational assessments evolved across the negotiation."""
    if not op_summaries:
        return 'No operational summaries were recorded during this session.'
    api_key = os.getenv('OPENAI_API_KEY')
    ordered = sorted(op_summaries, key=lambda x: int(x.get('idx', 0)))
    joined = '\n'.join(f"Turn {item.get('idx', 0) + 1}: {item.get('summary', '')}" for item in ordered if item.get('summary'))
    if not joined:
        return 'No operational summaries were recorded during this session.'

    if not api_key:
        return 'Operational summaries indicate shifting emotional intensity with intermittent convergence opportunities; uncertainty remained around cultural priors while risk signals fluctuated by turn.'

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You synthesize negotiation timeline analyses."},
            {"role": "user", "content": (
                "Summarize in 3-4 sentences how the dispute evolved over time from these per-turn operational summaries. "
                "Highlight shifts in emotions, IRP posture, and movement toward/away from agreement.\n\n" + joined
            )}
        ],
        "temperature": 0.2,
        "max_tokens": 180
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return 'Operational summaries indicate shifting emotional intensity with intermittent convergence opportunities; uncertainty remained around cultural priors while risk signals fluctuated by turn.'

def llm_executive_brief(turns, op_summaries, final_outcome, pareto):
    """Draft a two-paragraph executive brief using GPT-5.2 when available."""
    turns = turns or []
    op_summaries = op_summaries or []
    api_key = os.getenv('OPENAI_API_KEY')

    dialogue_excerpt = "\n".join(
        f"{t.get('speaker', 'Unknown')}: {t.get('text', '')}"
        for t in turns[-60:]
    )
    timeline_excerpt = "\n".join(
        f"Turn {int(item.get('idx', 0)) + 1}: {item.get('summary', '')}"
        for item in sorted(op_summaries, key=lambda x: int(x.get('idx', 0)))
        if item.get('summary')
    )

    pareto_set = {
        (
            p.get('refund_label'),
            p.get('buyer_review'),
            p.get('seller_review'),
            p.get('seller_apology'),
            p.get('buyer_apology'),
        )
        for p in (pareto or [])
    }
    is_pareto_efficient = False
    if final_outcome:
        sig = (
            final_outcome.get('refund_label'),
            final_outcome.get('buyer_review'),
            final_outcome.get('seller_review'),
            final_outcome.get('seller_apology'),
            final_outcome.get('buyer_apology'),
        )
        is_pareto_efficient = sig in pareto_set

    fallback = (
        "The dispute centered on compensation, review removal, and apology expectations, then evolved from positional demands toward partial trade-offs as each side tested leverage and legitimacy. "
        + ("It ended with a detectible agreement package that balanced face-saving and practical closure. " if final_outcome else "It ended without a clearly detectable final agreement in the closing turns. ")
        + "Across the timeline summaries, emotional intensity and pressure language varied, but convergence attempts appeared when parties shifted from accusations to concrete terms."
        + "\n\n"
        + "The process could likely have gone better with earlier reframing around shared interests, explicit option-building, and mediator checkpoints that separated rights/power claims from underlying needs. "
        + ("The final package appears Pareto efficient under the configured utility model. " if is_pareto_efficient else "The final package does not appear Pareto efficient under the configured utility model, suggesting unclaimed joint gains remained. ")
        + "Buyer and seller could each have improved outcomes by making contingent offers sooner, clarifying non-negotiables, and sequencing apologies/review actions with measurable commitments."
    )

    if not api_key:
        return fallback

    prompt = (
        "Write exactly two paragraphs for an executive brief.\n"
        "Paragraph 1: summarize the dispute, how it evolved, and how it ended.\n"
        "Paragraph 2: explain how it could have gone better, including mediator actions, buyer/seller alternatives, and whether the solution was Pareto efficient.\n"
        "Make the analysis insightful and specific, not generic. Use concrete dynamics from the timeline and dialogue.\n"
        "Target roughly 4-6 sentences per paragraph.\n"
        "Keep a neutral, executive tone focused on decision-relevant insights.\n"
        f"Detected final outcome: {json.dumps(final_outcome)}\n"
        f"Pareto efficient under configured utility model: {is_pareto_efficient}\n"
        f"Operational summaries over time:\n{timeline_excerpt or '[none]'}\n\n"
        f"Dialogue excerpt:\n{dialogue_excerpt or '[none]'}"
    )

    payload = {
        "model": "gpt-5.2",
        "messages": [
            {"role": "system", "content": "You draft executive mediation briefs for negotiation outcomes."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 450
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=18) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        content = (data['choices'][0]['message']['content'] or '').strip()
        return content or fallback
    except Exception:
        return fallback


def estimate_risk(turns_so_far):
    if not turns_so_far:
        return {'score':0,'label':'Low','negative_signals':0,'threats':0}
    total_neg    = sum(t['negative_signals'] for t in turns_so_far)
    total_threat = sum(t['threat_signals']   for t in turns_so_far)
    avg_anger    = float(np.mean([t['emotions'].get('anger',0) for t in turns_so_far]))
    score = min(1.0, total_neg*0.12 + total_threat*0.2 + avg_anger*0.6)
    label = 'Low' if score<0.33 else ('Medium' if score<0.66 else 'High')
    return {'score':round(float(score),3),'label':label,
            'negative_signals':int(total_neg),'threats':int(total_threat)}

def llm_intervention_assessment(turns_so_far, prior_intervention_turns=None):
    api_key = os.getenv('OPENAI_API_KEY')
    prior_intervention_turns = prior_intervention_turns or []
    reasons_allowed = {
        'Escalation of Conflict',
        'Impasse',
        'Miscommunication',
        'Unreasonable demands',
        'Invocation',
        'None',
    }

    if not turns_so_far:
        return {
            'rating': 1,
            'reason': 'None',
            'statement': 'I will continue to observe while you work toward a solution.',
            'should_intervene': False,
        }

    transcript = '\n'.join(
        f"Turn {i+1} - {t.get('speaker','Unknown')}: {t.get('text','')}"
        for i, t in enumerate(turns_so_far)
    )

    def fallback_assessment():
        risk = estimate_risk(turns_so_far)
        rating = 1
        reason = 'None'
        if risk['threats'] > 0:
            rating, reason = 5, 'Escalation of Conflict'
        elif risk['negative_signals'] >= 5:
            rating, reason = 4, 'Impasse'
        elif risk['score'] >= 0.45:
            rating, reason = 3, 'Miscommunication'
        statement = 'I recommend each side restate one concrete, feasible next step before continuing.' if rating >= 4 else 'Please continue negotiating directly; I will stay available if needed.'
        return {
            'rating': rating,
            'reason': reason,
            'statement': statement,
            'should_intervene': rating >= 4,
        }

    if not api_key:
        return fallback_assessment()

    prompt = """Imagine you are playing the role of a mediator in a buyer/seller purchase dispute. Your goal is to allow participants to resolve their dispute on their own if possible, but to intervene if necessary. Some reasons to intervene include:
1. Escalation of Conflict: if the conversation becomes heated with parties resorting to personal attacks or hostile language
2. Impasse: when parties reach a deadlock and are unable to move forward
3. Miscommunication: if there are signs that the parties are misunderstanding each other’s points
4. Unreasonable demands: If one party is making unreasonable demands that the other party can’t possibly meet.
5. Invocation: If one party asks the mediator to interject. 

You will be given the conversation so far. Rate the situation on a scale from 1 to 5 with 1 meaning definitely don’t intervene and 5 meaning definitely intervene. Provide (a) the rating on if to intervene, (b) the reason to intervene, from the list above, and (c) a one sentence statement you might tell the parties at this point. 

You do not need to intervene every turn, and should consider how recently you've intervened before making a decision.

Any score over a seven will trigger an intervention, and your message will be sent to both the buyer and seller."""

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nRecent intervention turns: {prior_intervention_turns}. Return JSON with fields rating, reason, statement."}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        content = data['choices'][0]['message']['content']
        obj = json.loads(content)
        rating = int(float(obj.get('rating', 1)))
        rating = max(1, min(5, rating))
        reason = str(obj.get('reason', 'None')).strip() or 'None'
        if reason not in reasons_allowed:
            reason = 'None'
        statement = str(obj.get('statement', '')).strip() or 'I will continue monitoring while you negotiate directly.'
        return {
            'rating': rating,
            'reason': reason,
            'statement': statement,
            'should_intervene': rating >= 4,
        }
    except Exception:
        return fallback_assessment()


def get_advisor(turns_so_far, current_turn):
    prior_intervention_turns = []
    for i, t in enumerate(turns_so_far[:-1]):
        adv = t.get('advisor') if isinstance(t, dict) else None
        if isinstance(adv, dict) and adv.get('should_intervene'):
            prior_intervention_turns.append(i + 1)
    assessment = llm_intervention_assessment(turns_so_far, prior_intervention_turns)
    return {
        'should_intervene': bool(assessment.get('should_intervene')),
        'action': 'Intervene' if assessment.get('should_intervene') else 'Observe',
        'rating': int(assessment.get('rating', 1)),
        'reason': assessment.get('reason', 'None'),
        'statement': assessment.get('statement', ''),
    }

# ── Country Prediction ────────────────────────────────────────────────────────
COUNTRY_SVG = {
    'United States': {'lat':37.09,'lng':-95.71,'flag':'🇺🇸'},
    'United Kingdom':{'lat':55.37,'lng':-3.43, 'flag':'🇬🇧'},
    'Germany':       {'lat':51.16,'lng':10.45, 'flag':'🇩🇪'},
    'China':         {'lat':35.86,'lng':104.19,'flag':'🇨🇳'},
    'Japan':         {'lat':36.20,'lng':138.25,'flag':'🇯🇵'},
    'India':         {'lat':20.59,'lng':78.96, 'flag':'🇮🇳'},
    'France':        {'lat':46.22,'lng':2.21,  'flag':'🇫🇷'},
    'Australia':     {'lat':-25.27,'lng':133.77,'flag':'🇦🇺'},
    'Canada':        {'lat':56.13,'lng':-106.34,'flag':'🇨🇦'},
    'Brazil':        {'lat':-14.23,'lng':-51.92,'flag':'🇧🇷'},
    'Mexico':        {'lat':23.63,'lng':-102.55,'flag':'🇲🇽'},
    'South Africa':  {'lat':-30.56,'lng':22.94,'flag':'🇿🇦'},
}


def _safe_build_predictor(language):
    try:
        return RegionPredictor(language=language, model_name='svm')
    except Exception:
        return None


PREDICTOR = _safe_build_predictor('english')
PREDICTOR_ZH = _safe_build_predictor('chinese')

COUNTRY_LABEL_MAP = {
    'U.S.':'United States','US':'United States','United States':'United States',
    'U.K.':'United Kingdom','UK':'United Kingdom','United Kingdom':'United Kingdom',
    'Mexico':'Mexico','South Africa':'South Africa','Germany':'Germany','China':'China',
    'Japan':'Japan','India':'India','France':'France','Australia':'Australia','Canada':'Canada','Brazil':'Brazil'
}

def _normalize_probabilities(probabilities):
    norm = {}
    if not isinstance(probabilities, dict):
        return norm
    for raw_label, raw_prob in probabilities.items():
        canonical = COUNTRY_LABEL_MAP.get(str(raw_label).strip(), str(raw_label).strip())
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            continue
        norm[canonical] = norm.get(canonical, 0.0) + max(0.0, prob)
    total = sum(norm.values())
    if total > 0:
        norm = {k: round(v / total, 6) for k, v in norm.items()}
    return norm


def _to_geo_payload(label, confidence, probabilities=None):
    canonical = COUNTRY_LABEL_MAP.get(label, label)
    info = COUNTRY_SVG.get(canonical, {'lat':0,'lng':0,'flag':'🌍'})
    return {
        'country': canonical,
        'confidence': round(float(confidence or 0),2),
        'lat': info['lat'],
        'lng': info['lng'],
        'flag': info['flag'],
        'probabilities': _normalize_probabilities(probabilities),
    }

def predict_country_with_model(turns, role):
    role_turns = [t.get('text', '').strip() for t in turns if t.get('speaker') == role and t.get('text', '').strip()]
    if not role_turns:
        return _to_geo_payload('United States', 0.0, {})
    try:
        if PREDICTOR is None:
            return predict_country(turns, role)
        outputs = PREDICTOR.predict_batch(role_turns)
        if not outputs:
            return _to_geo_payload('United States', 0.0, {})

        combined_probs = {}
        for out in outputs:
            probs = _normalize_probabilities(out.get('probabilities', {}))
            for label, prob in probs.items():
                combined_probs[label] = combined_probs.get(label, 0.0) + prob

        turn_count = len(outputs)
        averaged_probs = {label: (prob / turn_count) for label, prob in combined_probs.items()} if turn_count else {}
        averaged_probs = _normalize_probabilities(averaged_probs)

        if averaged_probs:
            pred, conf = max(averaged_probs.items(), key=lambda item: item[1])
        else:
            pred, conf = 'United States', 0.0
        return _to_geo_payload(pred, conf, averaged_probs)
    except Exception:
        return predict_country(turns, role)



def predict_cn_region_with_model(turns, role):
    role_turns = [t.get('text', '').strip() for t in turns if t.get('speaker') == role and t.get('text', '').strip()]
    base_province_probs = constrain_cn_probs_to_model_scope(infer_cn_province_distribution(turns, role))
    if not role_turns:
        return {
            'country': 'China',
            'confidence': 0.0,
            'probabilities': {'North': 0.25, 'Central': 0.25, 'Wu_Min': 0.25, 'Xian_Yue': 0.25},
            'province_probabilities': base_province_probs,
        }

    if PREDICTOR_ZH is None:
        # fallback to lightweight heuristic if chinese model is unavailable
        proxy = {
            'North': base_province_probs.get('beijing', 0) + base_province_probs.get('tianjin', 0) + base_province_probs.get('hebei', 0) + base_province_probs.get('shandong', 0),
            'Central': base_province_probs.get('henan', 0) + base_province_probs.get('hubei', 0) + base_province_probs.get('hunan', 0) + base_province_probs.get('sichuan', 0),
            'Wu_Min': base_province_probs.get('shanghai', 0) + base_province_probs.get('jiangsu', 0) + base_province_probs.get('zhejiang', 0) + base_province_probs.get('fujian', 0),
            'Xian_Yue': base_province_probs.get('guangdong', 0) + base_province_probs.get('guangxi', 0) + base_province_probs.get('hainan', 0) + base_province_probs.get('hong kong', 0),
        }
        total = sum(proxy.values()) or 1.0
        normalized = {k: round(v / total, 6) for k, v in proxy.items()}
        pred, conf = max(normalized.items(), key=lambda item: item[1])
        return {
            'country': 'China',
            'confidence': round(conf, 2),
            'probabilities': normalized,
            'region': pred,
            'province_probabilities': base_province_probs,
        }

    outputs = PREDICTOR_ZH.predict_batch(role_turns)
    combined = {}
    for out in outputs:
        for label, prob in (out.get('probabilities') or {}).items():
            combined[label] = combined.get(label, 0.0) + float(prob)
    count = len(outputs) or 1
    averaged = {k: v / count for k, v in combined.items()}
    total = sum(averaged.values()) or 1.0
    normalized = {k: round(v / total, 6) for k, v in averaged.items()}
    pred, conf = max(normalized.items(), key=lambda item: item[1])
    province_probs = project_region_probs_to_provinces(normalized, base_province_probs)
    return {
        'country': 'China',
        'confidence': round(conf, 2),
        'probabilities': normalized,
        'region': pred,
        'province_probabilities': province_probs,
    }

def predict_country(turns, role):
    text  = ' '.join(t['text'] for t in turns if t['speaker']==role).lower()
    formal = len(re.findall(r'\b(therefore|hence|furthermore|shall|hereby|pursuant|esteemed|aforesaid)\b', text))
    direct = len(re.findall(r"\b(bottom line|let's|gonna|want|deal|okay|yeah|sure|honestly|look)\b", text))
    polite = len(re.findall(r'\b(kindly|please|appreciate|grateful|honoured|humbly|sincerely)\b', text))
    if formal > direct and polite > 1:
        ranking = [('United Kingdom',0.40),('Germany',0.30),('Japan',0.18),('France',0.12)]
    elif direct > formal:
        ranking = [('United States',0.52),('Australia',0.20),('Canada',0.18),('Brazil',0.10)]
    elif polite > 2:
        ranking = [('Japan',0.42),('India',0.28),('China',0.20),('United Kingdom',0.10)]
    else:
        ranking = [('United States',0.38),('United Kingdom',0.27),('Germany',0.20),('China',0.15)]
    country, conf = ranking[0]
    info = COUNTRY_SVG.get(country, {'lat':0,'lng':0,'flag':'🌍'})
    return {
        'country': country,
        'confidence': round(conf, 2),
        'lat': info['lat'],
        'lng': info['lng'],
        'flag': info['flag'],
        'probabilities': _normalize_probabilities({label: score for label, score in ranking}),
    }

# ── Plotting ──────────────────────────────────────────────────────────────────
BG   = '#07090f'
CARD = '#0d1520'
GRID = '#1a2840'
TEXT = '#94a3b8'
EMO_COLORS = {
    'anger':'#f43f5e','fear':'#a78bfa','joy':'#22d3a5',
    'surprise':'#fbbf24','compassion':'#60a5fa','neutral':'#6b7280',
}

def plt_base(figsize=(7,4)):
    fig,ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values(): spine.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    return fig, ax

def fig_b64(fig):
    buf = io.BytesIO()
    plt.savefig(buf,format='png',dpi=140,bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_pareto_plot(outcomes, pareto, final_outcome=None, title='Solution Space'):
    fig,ax = plt_base((6.5,5))
    # All outcomes
    bx=[o['buyer_util'] for o in outcomes]; sy=[o['seller_util'] for o in outcomes]
    ax.scatter(bx,sy,color='#1e3a5f',alpha=0.5,s=25,zorder=1)
    # Pareto frontier
    if pareto:
        px=[p['buyer_util'] for p in pareto]; py=[p['seller_util'] for p in pareto]
        ax.plot(px,py,color='#22d3a5',linewidth=2.5,label='Pareto Frontier',zorder=3,marker='o',markersize=5)
        ax.fill_between(px,py,alpha=0.08,color='#22d3a5')
    # Final agreement
    if final_outcome:
        ax.scatter([final_outcome['buyer_util']],[final_outcome['seller_util']],
                   color='#f59e0b',s=250,zorder=6,marker='*',
                   label=f"Agreement (B:{final_outcome['buyer_util']:.0f}, S:{final_outcome['seller_util']:.0f})")
    ax.set_xlabel('Buyer Utility', color=TEXT, fontsize=9)
    ax.set_ylabel('Seller Utility', color=TEXT, fontsize=9)
    ax.set_title(title, color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(-5,105); ax.set_ylim(-5,105)
    leg = ax.legend(fontsize=8,facecolor='#1a2840',edgecolor=GRID,labelcolor=TEXT)
    return fig_b64(fig)

def make_emotion_plot(turns, dimension='joy'):
    fig,ax = plt_base((8,3.5))
    buyer  = [(t['idx'],t['emotions'].get(dimension,0)) for t in turns if t['speaker']=='Buyer']
    seller = [(t['idx'],t['emotions'].get(dimension,0)) for t in turns if t['speaker']=='Seller']
    color  = EMO_COLORS.get(dimension,'#94a3b8')
    # Buyer - solid line
    if buyer:
        bx,by=zip(*buyer)
        ax.plot(bx,by,color=color,marker='o',markersize=5,linewidth=2.2,label='Buyer',zorder=3,alpha=0.95)
        ax.fill_between(bx,by,alpha=0.12,color=color)
    # Seller - dashed
    if seller:
        sx,sy2=zip(*seller)
        ax.plot(sx,sy2,color=color,marker='s',markersize=5,linewidth=2.2,
                label='Seller',zorder=3,linestyle='--',alpha=0.7)
    ax.set_xlabel('Turn Index',color=TEXT,fontsize=9)
    ax.set_ylabel('Intensity',color=TEXT,fontsize=9)
    ax.set_title(f'{dimension.capitalize()} Over Time',color='#e2e8f0',fontsize=11,fontweight='bold',pad=10)
    ax.set_ylim(-0.05,1.1)
    ax.legend(fontsize=8,facecolor='#1a2840',edgecolor=GRID,labelcolor=TEXT)
    return fig_b64(fig)

def make_all_emotions_plot(turns):
    """Multi-line chart showing all emotions for both roles."""
    fig,axes = plt.subplots(2,1,figsize=(8,5),facecolor=BG)
    for ax,role,color_base in zip(axes,['Buyer','Seller'],['solid','dashed']):
        ax.set_facecolor(CARD)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)
        ax.tick_params(colors=TEXT,labelsize=7)
        ax.grid(True,color=GRID,linewidth=0.5,alpha=0.6)
        rturn = [(t['idx'],t['emotions']) for t in turns if t['speaker']==role]
        if rturn:
            xs = [r[0] for r in rturn]
            for emo,c in EMO_COLORS.items():
                ys = [r[1].get(emo,0) for r in rturn]
                ls = '--' if role=='Seller' else '-'
                ax.plot(xs,ys,color=c,linewidth=1.8,label=emo.capitalize(),linestyle=ls,marker='o' if len(xs)<12 else None,markersize=3)
        ax.set_ylabel(role,color=TEXT,fontsize=8)
        ax.set_ylim(-0.05,1.05)
    axes[0].legend(fontsize=7,facecolor='#1a2840',edgecolor=GRID,labelcolor=TEXT,ncol=3,loc='upper right')
    axes[1].set_xlabel('Turn Index',color=TEXT,fontsize=8)
    plt.suptitle('All Emotions Over Time',color='#e2e8f0',fontsize=11,fontweight='bold',y=1.01)
    plt.tight_layout()
    return fig_b64(fig)



def extract_uploaded_bundle_metadata(path, ext):
    meta = {}
    if ext != 'json':
        return meta
    try:
        with open(path, encoding='utf-8-sig', errors='replace') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if isinstance(data.get('op_summaries'), list):
                meta['op_summaries'] = data.get('op_summaries')
            if isinstance(data.get('step_cache'), dict):
                meta['step_cache'] = data.get('step_cache')
            if isinstance(data.get('country'), dict):
                meta['country'] = data.get('country')
            if isinstance(data.get('language'), str):
                val = data.get('language', '').strip().upper()
                if val in {'EN', 'CN'}:
                    meta['language'] = val
    except Exception:
        return meta
    return meta


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('upload.html')

@app.route('/pre')
def pre(): return render_template('pre.html')

@app.route('/negotiate')
def negotiate(): return render_template('negotiate.html')

@app.route('/post')
def post(): return render_template('post.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    f = request.files['file']
    if not f or not allowed(f.filename): return jsonify({'error':'Invalid file type'}),400
    filename = secure_filename(f.filename)
    ext = filename.rsplit('.',1)[1].lower()
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    try:
        turns = parse_file(path, ext)
        if not turns: return jsonify({'error':'No turns found in file'}),400

        bundle_meta = extract_uploaded_bundle_metadata(path, ext)
        language = bundle_meta.get('language', extract_dialogue_language(path, ext))
        has_cached = turns_have_cached_enrichment(turns)
        if not has_cached:
            turns = enrich(turns)

        if language == 'CN':
            for t in turns:
                txt = t.get('text', '')
                if txt and not t.get('translation'):
                    t['translation'] = llm_translate_cn_to_en(txt)

        pre_justifications = extract_pre_dispute_justifications(path, ext)
        bw, sw = extract_preference_weights(path, ext)
        calc_bw = {**DEFAULT_BUYER_WEIGHTS, **bw}
        calc_sw = {**DEFAULT_SELLER_WEIGHTS, **sw}
        outcomes = generate_all_outcomes(calc_bw, calc_sw)
        pareto   = compute_pareto(outcomes)
        pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: KODIS Solution Space')

        if language == 'CN':
            buyer_c = predict_cn_region_with_model(turns, 'Buyer')
            seller_c = predict_cn_region_with_model(turns, 'Seller')
            empty_probs = {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()}
            buyer_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='buyer')
            seller_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='seller')
        else:
            buyer_c  = predict_country_with_model(turns,'Buyer')
            seller_c = predict_country_with_model(turns,'Seller')

        return jsonify({
            'turns':turns,'filename':filename,
            'language': language,
            'buyer_weights':bw,'seller_weights':sw,
            'outcomes':outcomes,'pareto':pareto,
            'pareto_img':pareto_img,
            'country':{'buyer':buyer_c,'seller':seller_c},
            'pre_justifications': pre_justifications,
            'op_summaries': bundle_meta.get('op_summaries', []),
            'step_cache': bundle_meta.get('step_cache', {}),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500

@app.route('/api/update_weights', methods=['POST'])
def api_update_weights():
    data = request.json
    bw   = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw   = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    outcomes = generate_all_outcomes(bw, sw)
    pareto   = compute_pareto(outcomes)
    pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: KODIS Solution Space')
    return jsonify({'outcomes':outcomes,'pareto':pareto,'pareto_img':pareto_img})

@app.route('/api/step', methods=['POST'])
def api_step():
    data  = request.json
    idx   = data.get('idx',0)
    turns = data.get('turns',[])
    dim   = data.get('emotion_dim','joy')
    language = str(data.get('language', 'EN')).upper()
    if not turns or idx >= len(turns): return jsonify({'error':'Invalid'}),400

    cur = turns[idx]
    if not cur.get('emotions'):
        cur['emotions'] = llm_emotion_scores(cur.get('text',''))
    if language == 'CN' and cur.get('text') and not cur.get('translation'):
        cur['translation'] = llm_translate_cn_to_en(cur.get('text',''))
    turns_so_far = turns[:idx+1]

    risk  = estimate_risk(turns_so_far)
    adv   = cur.get('advisor') or get_advisor(turns_so_far, cur)
    cur['advisor'] = adv
    emo_img = make_emotion_plot(turns_so_far, dim)
    if language == 'CN':
        buyer_c = predict_cn_region_with_model(turns_so_far, 'Buyer')
        seller_c = predict_cn_region_with_model(turns_so_far, 'Seller')
        buyer_c['province_map_b64'] = make_cn_province_map(
            buyer_c.get('province_probabilities', {}), role='buyer', region_probs=buyer_c.get('probabilities', {})
        )
        seller_c['province_map_b64'] = make_cn_province_map(
            seller_c.get('province_probabilities', {}), role='seller', region_probs=seller_c.get('probabilities', {})
        )
    else:
        buyer_c = predict_country_with_model(turns_so_far, 'Buyer')
        seller_c = predict_country_with_model(turns_so_far, 'Seller')
    country_snapshot = {'buyer':buyer_c,'seller':seller_c}
    op_summary = llm_operational_summary(turns_so_far, country_snapshot, risk)
    return jsonify({
        'risk':risk,
        'advisor':adv,
        'emotion_img':emo_img,
        'current_turn':cur,
        'country':country_snapshot,
        'operational_summary': op_summary,
    })

@app.route('/api/post_summary', methods=['POST'])
def api_post_summary():
    data = request.json or {}
    turns = data.get('turns', [])
    bw    = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw    = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    op_summaries = data.get('op_summaries', [])
    final_outcome = data.get('final_outcome')

    turns_enriched = enrich(turns) if not turns_have_cached_enrichment(turns) else turns
    risk = estimate_risk(turns_enriched)
    outcomes = generate_all_outcomes(bw, sw)
    pareto = compute_pareto(outcomes)

    match = None
    if final_outcome:
        match = next((o for o in outcomes if
                      o['refund_label']   == final_outcome.get('refund_label') and
                      o['buyer_review']   == final_outcome.get('buyer_review') and
                      o['seller_review']  == final_outcome.get('seller_review') and
                      o['seller_apology'] == final_outcome.get('seller_apology') and
                      o['buyer_apology']  == final_outcome.get('buyer_apology')), None)

    post_img = make_pareto_plot(outcomes, pareto, match, title='Post-Negotiation: Solution Space')
    executive_brief = llm_executive_brief(turns_enriched, op_summaries, match, pareto)
    return jsonify({
        'risk': risk,
        'post_img': post_img,
        'final_outcome': match,
        'outcomes': outcomes,
        'pareto': pareto,
        'executive_brief': executive_brief,
    })

@app.route('/api/export_pdf', methods=['POST'])
def api_export_pdf():
    data         = request.json
    turns        = data.get('turns',[])
    filename     = data.get('filename','negotiation')
    pre_b64      = data.get('pre_img','')
    emo_b64      = data.get('emotion_img','')
    post_b64     = data.get('post_img','')
    final_outcome= data.get('final_outcome', None)
    bw           = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw           = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    op_summaries = data.get('op_summaries', [])

    # Generate post-negotiation pareto if not provided
    outcomes = generate_all_outcomes(bw, sw)
    if final_outcome and ('buyer_util' not in final_outcome or 'seller_util' not in final_outcome):
        final_outcome = next((o for o in outcomes if
                              o['refund_label']   == final_outcome.get('refund_label') and
                              o['buyer_review']   == final_outcome.get('buyer_review') and
                              o['seller_review']  == final_outcome.get('seller_review') and
                              o['seller_apology'] == final_outcome.get('seller_apology') and
                              o['buyer_apology']  == final_outcome.get('buyer_apology')), final_outcome)

    if not post_b64:
        pareto   = compute_pareto(outcomes)
        fo_dict  = final_outcome if final_outcome else None
        post_b64 = make_pareto_plot(outcomes, pareto, fo_dict, title='Post-Negotiation Solution Space')

    # Generate all-emotions chart
    turns_enriched = enrich(turns) if not turns_have_cached_enrichment(turns) else turns
    all_emo_b64 = make_all_emotions_plot(turns_enriched)
    if not emo_b64: emo_b64 = all_emo_b64

    risk = estimate_risk(turns_enriched)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.7*inch, rightMargin=0.7*inch,
                            topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle('T',parent=styles['Title'],textColor=colors.HexColor('#0f172a'),fontSize=20,spaceAfter=6)
    h2_s    = ParagraphStyle('H2',parent=styles['Heading2'],textColor=colors.HexColor('#1e3a5f'),fontSize=12,spaceAfter=4)
    body_s  = ParagraphStyle('B',parent=styles['Normal'],fontSize=8.8,spaceAfter=3,leading=12)
    mono_s  = ParagraphStyle('M',parent=styles['Normal'],fontSize=8,fontName='Courier',spaceAfter=2,leading=11)

    story = []
    cover = Table([[
      Paragraph('<b>NegotiationLens — Mission Summary Report</b>', ParagraphStyle('ct', parent=styles['Normal'], textColor=colors.white, fontSize=14, leading=16)),
      Paragraph(f'<b>Risk:</b> {risk["label"]} ({risk["score"]*100:.0f}%)<br/><b>Turns:</b> {len(turns)}', ParagraphStyle('cr', parent=styles['Normal'], textColor=colors.white, fontSize=9, leading=12))
    ]], colWidths=[4.5*inch, 1.7*inch])
    cover.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#1e3a5f')),
      ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#0f172a')),
      ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
      ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
      ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
    ]))
    story.append(cover)
    story.append(Spacer(1,0.12*inch))
    story.append(Paragraph(f'Source file: <b>{filename}</b>', body_s))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#94a3b8')))
    story.append(Spacer(1,0.08*inch))
    agreement = 'Agreement detected' if final_outcome else 'No definitive agreement detected'
    story.append(Paragraph(f'<b>Executive Brief:</b> {agreement}. Risk posture is <b>{risk["label"]}</b> with <b>{risk["negative_signals"]}</b> negative signals and <b>{risk["threats"]}</b> threats.', body_s))
    story.append(Spacer(1,0.08*inch))

    evolution_summary = llm_evolution_summary(op_summaries)
    story.append(Paragraph('Dispute Evolution (Operational Timeline)', h2_s))
    story.append(Paragraph(evolution_summary, body_s))
    story.append(Spacer(1,0.08*inch))

    # Weights table
    story.append(Paragraph('KODIS Preference Weights', h2_s))
    issues = ['refund','buyer_review','seller_review','seller_apology','buyer_apology']
    labels = ['Refund','Buyer Review Removed','Seller Review Removed','Seller Apologizes','Buyer Apologizes']
    w_data = [['Issue','Buyer Weight','Seller Weight']] + [[l, str(bw.get(i,0)), str(sw.get(i,0))] for l,i in zip(labels,issues)]
    wt = Table(w_data, colWidths=[2.8*inch,1.8*inch,1.8*inch])
    wt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e3a5f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f4fa'),colors.white]),
    ]))
    story.append(wt); story.append(Spacer(1,0.12*inch))

    # Risk
    story.append(Paragraph('Risk Summary', h2_s))
    r_data = [['Risk Score','Risk Level','Negative Signals','Threats'],
              [f"{risk['score']*100:.0f}%", risk['label'], str(risk['negative_signals']), str(risk['threats'])]]
    rt = Table(r_data, colWidths=[1.6*inch]*4)
    rt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e3a5f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#f0f4fa')),
    ]))
    story.append(rt); story.append(Spacer(1,0.12*inch))

    # Final agreement
    if final_outcome:
        bu = final_outcome.get('buyer_util')
        su = final_outcome.get('seller_util')
        story.append(Paragraph('Final Agreed Outcome', h2_s))
        fa_items = [
            ['Refund', final_outcome.get('refund_label','—')],
            ['Buyer Review Removed', 'Yes' if final_outcome.get('buyer_review') else 'No'],
            ['Seller Review Removed','Yes' if final_outcome.get('seller_review') else 'No'],
            ['Seller Apologizes',    'Yes' if final_outcome.get('seller_apology') else 'No'],
            ['Buyer Apologizes',     'Yes' if final_outcome.get('buyer_apology') else 'No'],
            ['Buyer Utility',        (f"{bu:.0f}/100" if isinstance(bu, (int, float)) else '—')],
            ['Seller Utility',       (f"{su:.0f}/100" if isinstance(su, (int, float)) else '—')],
        ]
        fat = Table([['Issue','Value']] + fa_items, colWidths=[2.8*inch, 3.6*inch])
        fat.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#22d3a5')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#07090f')),
            ('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0faf7'),colors.white]),
        ]))
        story.append(fat); story.append(Spacer(1,0.12*inch))

    def add_img(b64s, label, w=6.2):
        if not b64s: return
        story.append(Paragraph(label, h2_s))
        ib = io.BytesIO(base64.b64decode(b64s))
        story.append(RLImage(ib, width=w*inch, height=w*0.6*inch))
        story.append(Spacer(1,0.1*inch))

    add_img(pre_b64,  'Pre-Negotiation: KODIS Solution Space & Pareto Frontier')
    add_img(emo_b64,  'Emotion Trajectory (All Emotions)')
    add_img(post_b64, 'Post-Negotiation: Solution Space with Agreement Marked')

    story.append(Paragraph('Full Transcript', h2_s))
    for t in turns:
        story.append(Paragraph(f"<b>[{t['speaker']}]</b>  {t['text'][:500]}", body_s))
    story.append(Spacer(1,0.08*inch))
    story.append(Paragraph('<i>Generated by NegotiationLens · KODIS Dispute Analysis</i>', body_s))
    doc.build(story)
    buf.seek(0)
    return send_file(buf,mimetype='application/pdf',as_attachment=True,download_name='negotiation_report.pdf')



@app.route('/api/export_enriched', methods=['POST'])
def api_export_enriched():
    data = request.json or {}
    turns = data.get('turns', [])
    payload = {
        'language': str(data.get('language', 'EN')).upper() if data.get('language') else 'EN',
        'buyer_weights': data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS),
        'seller_weights': data.get('seller_weights', DEFAULT_SELLER_WEIGHTS),
        'pre_dispute_justifications': data.get('pre_justifications', {}),
        'turns': turns,
        'op_summaries': data.get('op_summaries', []),
        'final_outcome': data.get('final_outcome'),
        'country': data.get('country', {}),
        'step_cache': data.get('step_cache', {}),
    }
    buf = io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='application/json', as_attachment=True, download_name='negotiation_enriched.json')

if __name__=='__main__':
    os.makedirs('uploads',exist_ok=True)
    app.run(debug=True,port=5000)
