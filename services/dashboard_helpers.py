import os, json, io, re, base64
import urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from predictor import RegionPredictor
from services.llm_client import openai_chat_json, openai_chat_text

NEG_SIGNALS = ['walkaway','walk away','final','no deal','reject','refuse','unacceptable','sue','lawyer','court','legal action','worst','scam']

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
    try:
        return openai_chat_text(payload, timeout=60, api_key=api_key)
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


def _canonicalize_geo_name(raw):
    s = str(raw or '').strip().lower()
    if not s:
        return ''
    for ch in ['_', '-', '.', '·', '•', '，', ',']:
        s = s.replace(ch, ' ')
    s = ' '.join(s.split())
    # English administrative suffixes
    suffixes = [
        ' province', ' autonomous region', ' municipality', ' city', ' region',
        ' special administrative region'
    ]
    for suf in suffixes:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    # Chinese administrative suffixes
    for suf in ['省', '市', '自治区', '特别行政区']:
        s = s.replace(suf, '')
    return s.strip()


CN_PROVINCE_KEYWORDS = {
    'beijing': ['beijing', '北京'],
    'tianjin': ['tianjin', '天津'],
    'hebei': ['hebei', '河北'],
    'shanxi': ['shanxi', '山西'],
    'inner mongolia': ['inner mongolia', 'neimenggu', '内蒙古'],
    'liaoning': ['liaoning', '辽宁'],
    'jilin': ['jilin', '吉林'],
    'heilongjiang': ['heilongjiang', '黑龙江'],
    'shanghai': ['shanghai', '上海'],
    'jiangsu': ['jiangsu', '江苏'],
    'zhejiang': ['zhejiang', '浙江'],
    'anhui': ['anhui', '安徽'],
    'fujian': ['fujian', '福建'],
    'jiangxi': ['jiangxi', '江西'],
    'shandong': ['shandong', '山东'],
    'henan': ['henan', '河南'],
    'hubei': ['hubei', '湖北'],
    'hunan': ['hunan', '湖南'],
    'guangdong': ['guangdong', '广东'],
    'guangxi': ['guangxi', '广西'],
    'hainan': ['hainan', '海南'],
    'chongqing': ['chongqing', '重庆'],
    'sichuan': ['sichuan', '四川'],
    'guizhou': ['guizhou', '贵州'],
    'yunnan': ['yunnan', '云南'],
    'tibet': ['tibet', 'xizang', '西藏'],
    'shaanxi': ['shaanxi', '陕西'],
    'gansu': ['gansu', '甘肃'],
    'qinghai': ['qinghai', '青海'],
    'ningxia': ['ningxia', '宁夏'],
    'xinjiang': ['xinjiang', '新疆'],
    'hong kong': ['hong kong', '香港'],
    'macau': ['macau', '澳门'],
    'taiwan': ['taiwan', '台湾'],
}


def _normalize_cn_province_name(raw):
    key = _canonicalize_geo_name(raw)
    if not key:
        return None
    direct = CN_PROVINCE_ALIASES.get(key, CN_PROVINCE_ALIASES.get(key.lower()))
    if direct:
        return direct
    for prov, kws in CN_PROVINCE_KEYWORDS.items():
        for kw in kws:
            if _canonicalize_geo_name(kw) in key or key in _canonicalize_geo_name(kw):
                return prov
    return None


def _extract_name_from_feature(feature):
    props = feature.get('properties', {}) if isinstance(feature, dict) else {}
    for key in ['name', 'NAME_1', 'NL_NAME_1', 'province', 'prov_name', 'NAME_CHN', 'name_zh', 'NAME', 'NAME_EN', 'ENGTYPE_1', 'admin']:
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
    # Grouping requested by user.
    'Wu_Min': ['jiangsu', 'shanghai', 'zhejiang', 'fujian', 'anhui'],
    'Xian_Yue': ['guangdong', 'jiangxi', 'hainan'],
    'Central': ['hubei', 'sichuan', 'hebei', 'hunan', 'henan', 'guizhou', 'chongqing', 'yunnan'],
    'North': ['beijing', 'shandong', 'liaoning', 'heilongjiang', 'jilin', 'tianjin', 'gansu', 'shaanxi', 'qinghai', 'inner mongolia', 'xinjiang', 'shanxi', 'ningxia'],
}

CN_REGION_LABEL_POS = {
    'North': (116.0, 39.2),
    'Central': (112.8, 31.5),
    'Wu_Min': (120.2, 30.2),
    'Xian_Yue': (113.2, 23.0),
}


def _hex_to_rgb(hex_color):
    h = str(hex_color).lstrip('#')
    if len(h) != 6:
        return (128, 128, 128)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


def _lerp_color(hex_lo, hex_hi, t):
    t = max(0.0, min(1.0, float(t)))
    lo = _hex_to_rgb(hex_lo)
    hi = _hex_to_rgb(hex_hi)
    return _rgb_to_hex(tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3)))


def _region_center(region_name):
    provinces = CN_REGION_PROVINCES.get(region_name, [])
    pts = [CN_PROVINCE_CENTROIDS.get(p) for p in provinces if CN_PROVINCE_CENTROIDS.get(p)]
    if not pts:
        return CN_REGION_LABEL_POS.get(region_name, (110.0, 32.0))
    lon = sum(p[0] for p in pts) / len(pts)
    lat = sum(p[1] for p in pts) / len(pts)
    return lon, lat


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
        return scoped
    return {k: round(v / total, 6) for k, v in scoped.items()}


def make_cn_province_map(province_probs, role='buyer', region_probs=None):
    probs = {k: float(v) for k, v in (province_probs or {}).items() if k in CN_PROVINCE_CENTROIDS}
    if not probs:
        return None
    total = sum(probs.values()) or 1.0
    probs = {k: v / total for k, v in probs.items()}

    bg = '#07111f'
    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor=bg)
    ax.set_facecolor(bg)
    features = _load_china_geojson_features()
    drew_geojson = False
    min_lon, max_lon, min_lat, max_lat = 200, -200, 90, -90

    if str(role).lower() == 'buyer':
        light_color = '#dbeafe'
        dark_color = '#1e3a8a'
    else:
        light_color = '#fee2e2'
        dark_color = '#7f1d1d'

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
        # Continuous scale: higher probability => darker color.
        return _lerp_color(light_color, dark_color, p)

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

        for region in CN_REGION_PROVINCES.keys():
            p = float(rp.get(region, 0.0))
            if p <= 0:
                continue
            lon, lat = _region_center(region)
            ax.text(
                lon, lat, f"{region} {p * 100:.1f}%",
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
            colors.append(color_for_prob(p))

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

def _unavailable_emotions():
    return {}

def enrich_with_llm(turns, language='EN'):
    """LLM-first enrichment (no heuristic emotion scoring)."""
    out = turns if isinstance(turns, list) else []
    for t in out:
        if not isinstance(t, dict):
            continue
        txt = t.get('text', '')
        if str(language).upper() == 'CN':
            if txt and not t.get('translation'):
                t['translation'] = llm_translate_cn_to_en(txt)
            emo_text = t.get('translation') or txt
        else:
            emo_text = txt
        t['emotions'] = llm_emotion_scores(emo_text)
        tl = str(emo_text or '').lower()
        t['negative_signals'] = sum(1 for s in NEG_SIGNALS if s in tl)
        t['threat_signals'] = sum(1 for s in ['sue', 'lawyer', 'court', 'legal action'] if s in tl)
    return out




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
    """Use GPT-4o emotion classification; return no scores when the LLM is unavailable."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return _unavailable_emotions()

    system_prompt = """You are a good emotion classification tool. Your task is to classify the emotion of the last speaker based on the contextual dialogue.
            Your output should be a JSON object with an 'emotion' field, categorizing the dialogue with a score for each: joy, anger, fear, sadness, surprise, compassion, or neutral. These scores should sum to one. If an utterance is neutral, then neutral must be one with everything other label set to zero.
            Here are a few examples of proper annotations:
            {"statement": "Hi! I'd like to return my jersey.", "emotion": {"joy": "0", "anger": "0", "fear": "0", "sadness": "0", "surprise": "0", "compassion": "0", "neutral": "1"}},
            {"statement": "Please understand this was for my dear nephew he loves Kobe. I understand we had a misunderstanding, last thing I want is to hurt your business. Let's resolve this together", "emotion": {"joy": "0", "anger": "0", "fear": "0.4", "sadness": "0", "surprise": "0", "compassion": "0.6", "neutral": "0"}},
            {"statement": "Thank you!", "emotion": {"joy": "1", "anger": "0", "fear": "0", "sadness": "0", "surprise": "0", "compassion": "0", "neutral": "0"}},
            {"statement": "I will report you to authorities for doing this.", "emotion": {"joy": "0", "anger": "1", "fear": "0", "sadness": "0", "surprise": "0", "compassion": "0", "neutral": "0" """

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"This is the context: {text}. What is the emotion of the current speaker?"}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    # print(payload)
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
        with urllib.request.urlopen(req, timeout=60) as resp:
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
            return _unavailable_emotions()
        out = {k: round(v / total, 3) for k, v in out.items()}
        if out['neutral'] >= 0.999:
            out = {k: 0.0 for k in out}
            out['neutral'] = 1.0

        neg = out['anger'] + out['fear'] + out['sadness']
        pos = out['joy'] + out['compassion']
        out['valence'] = round(max(-1, min(1, pos - neg)), 3)
        return out
    except Exception:
        return _unavailable_emotions()

def llm_detect_agreement_last_two(turns):
    """Detect whether the latest turns form an agreement and extract issue terms via LLM."""
    recent_turns = turns[-6:] if isinstance(turns, list) else []
    if len(recent_turns) < 2:
        return None
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None

    excerpt = "\n".join(f"{t.get('speaker','Unknown')}: {t.get('text','')}" for t in recent_turns)
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Extract only explicit final agreement from the latest negotiation turns. Respond with JSON only."},
            {"role": "user", "content": (
                "Decide whether the latest turns explicitly confirm a final settlement package. "
                "If not explicit, set agreed=false and outcome=null.\n"
                "If explicit, return agreed=true and outcome with:\n"
                "refund_label: Full|Half|None,\n"
                "buyer_review: 0|1 (1 means the BUYER'S review is removed/retracted),\n"
                "seller_review: 0|1 (1 means the SELLER'S review is removed/retracted),\n"
                "seller_apology: 0|1 (1 means seller gives an apology),\n"
                "buyer_apology: 0|1 (1 means buyer gives an apology).\n\n"
                "Review removals include remove/retract/withdraw/take down/pull down/delete.\n"
                "Do NOT set buyer_review or seller_review when those actions are negated (e.g., do not retract, won't remove).\n"
                "Only use terms that are explicitly accepted in the latest confirmed package.\n\n"
                f"Messages:\n{excerpt}\n\n"
                "JSON format: {\"agreed\":true|false,\"outcome\":{...}|null}"
            )}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(
        'https://api.openai.com/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        obj = json.loads(data['choices'][0]['message']['content'] or '{}')
        if not obj.get('agreed'):
            return None
        out = obj.get('outcome') if isinstance(obj.get('outcome'), dict) else {}
        refund_label = str(out.get('refund_label', 'None')).strip().capitalize()
        if refund_label not in {'Full', 'Half', 'None'}:
            refund_label = 'None'

        def _to_flag(v):
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, (int, float)):
                return 1 if int(v) else 0
            s = str(v).strip().lower()
            return 1 if s in {'1', 'yes', 'true', 'y'} else 0

        result = {
            'refund_label': refund_label,
            'refund': {'Full': 1.0, 'Half': 0.5, 'None': 0.0}[refund_label],
            'buyer_review': _to_flag(out.get('buyer_review', 0)),
            'seller_review': _to_flag(out.get('seller_review', 0)),
            'seller_apology': _to_flag(out.get('seller_apology', 0)),
            'buyer_apology': _to_flag(out.get('buyer_apology', 0)),
        }
        return result
    except Exception:
        return None


def _irp_patterns(turns_so_far):
    labels = ['Interest', 'Right', 'Power', 'Unavailable']
    total = {k: 0 for k in labels}
    by_speaker = {'buyer': {k: 0 for k in labels}, 'seller': {k: 0 for k in labels}}
    timeline = []
    for t in turns_so_far or []:
        speaker = str(t.get('speaker', '')).strip().lower()
        raw = (t.get('meta', {}) or {}).get('irp_label') or t.get('irp') or 'Unavailable'
        v = str(raw).strip().lower()
        label = 'Unavailable'
        if v.startswith('interest'):
            label = 'Interest'
        elif v.startswith('right'):
            label = 'Right'
        elif v.startswith('power'):
            label = 'Power'
        total[label] += 1
        if speaker == 'buyer':
            by_speaker['buyer'][label] += 1
        elif speaker == 'seller':
            by_speaker['seller'][label] += 1
        timeline.append(label)
    return {
        'total_counts': total,
        'buyer_counts': by_speaker['buyer'],
        'seller_counts': by_speaker['seller'],
        'recent_sequence': timeline[-8:],
    }


def llm_operational_summary(turns_so_far, country_snapshot, risk_snapshot, irp_patterns=None):
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
        return "LLM operational summary unavailable (missing OPENAI_API_KEY)."

    prompt = (
        "Write an operational summary in at most TWO sentences. "
        "Only highlight the most important facets of the dialogue so far. "
        "Use signals from: integrative potential, dialogue history progression, emotional trajectory, and IRP balance (Interests/Rights/Power). "
        "Keep it high-level, practical, and easy to understand for mediators. "
        "Treat country predictions as uncertain priors (not facts). "
        "Respond in English only.\n\n"
        f"Risk snapshot: {json.dumps(risk_snapshot)}\n"
        # f"Country snapshot: {json.dumps(country_snapshot)}\n"
        f"IRP patterns: {json.dumps(irp_patterns or {})}\n"
        f"Observed turns:\n{transcript_excerpt}"
    )
    print(prompt)
    payload = {
        "model": "gpt-4o",
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return "LLM operational summary unavailable (API request failed)."


def llm_irp_label(turns_so_far, current_turn):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return 'Unavailable'
    excerpt = "\n".join(f"{t.get('speaker','Unknown')}: {t.get('text','')}" for t in turns_so_far[-8:])
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Classify negotiation utterances into Interest, Right, or Power. Respond in English JSON only."},
            {"role": "user", "content": (
                "Label ONLY the final utterance in this context as one of: Interest, Right, Power. "
                "Return JSON: {\"irp_label\":\"Interest|Right|Power\"}.\n\n"
                f"{excerpt}"
            )}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    try:
        obj = openai_chat_json(payload, timeout=12, api_key=api_key)
        v = str(obj.get('irp_label', '')).strip().lower()
        if v.startswith('interest'): return 'Interest'
        if v.startswith('right'): return 'Right'
        if v.startswith('power'): return 'Power'
    except Exception:
        return 'Unavailable'
    return 'Unavailable'


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
        return 'LLM evolution summary unavailable (missing OPENAI_API_KEY).'

    payload = {
        "model": "gpt-4o",
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
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return (data['choices'][0]['message']['content'] or '').strip()
    except Exception:
        return 'LLM evolution summary unavailable (API request failed).'

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


    if not api_key:
        return 'LLM executive brief unavailable (missing OPENAI_API_KEY).'

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
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        content = (data['choices'][0]['message']['content'] or '').strip()
        return content or 'LLM executive brief unavailable (empty API response).'
    except Exception:
        return 'LLM executive brief unavailable (API request failed).'


def _unavailable_risk(reason):
    return {
        'score': 0.0,
        'risk_0_100': 0,
        'label': 'Unavailable',
        'negative_signals': 0,
        'threats': 0,
        'rationale_short': reason,
    }

def llm_failure_risk(turns_so_far):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return _unavailable_risk('LLM risk assessment unavailable (missing OPENAI_API_KEY).')
    if not turns_so_far:
        return {'score':0,'risk_0_100':0,'label':'Low','negative_signals':0,'threats':0}
    transcript = '\n'.join(
        f"Turn {i+1} - {t.get('speaker','Unknown')}: {t.get('text','')}"
        for i, t in enumerate(turns_so_far[-35:])
    )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Assess negotiation failure risk. Return ENGLISH JSON only."},
            {"role": "user", "content": (
                "Output JSON with fields: risk_0_100 (0-100 integer), label (Low|Medium|High), rationale_short (<=18 words).\n\n"
                f"{transcript}"
            )}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
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
        obj = json.loads(data['choices'][0]['message']['content'] or '{}')
        risk_100 = max(0, min(100, int(float(obj.get('risk_0_100', 0)))))
        label = str(obj.get('label', '')).strip().capitalize()
        if label not in {'Low', 'Medium', 'High'}:
            label = 'Unavailable'
        return {
            'score': round(risk_100 / 100.0, 3),
            'risk_0_100': risk_100,
            'label': label,
            'negative_signals': 0,
            'threats': 0,
            'rationale_short': str(obj.get('rationale_short', '')).strip(),
        }
    except Exception:
        return _unavailable_risk('LLM risk assessment unavailable (API request failed).')

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

    def unavailable_assessment(reason):
        return {
            'rating': 1,
            'reason': 'Unavailable',
            'statement': reason,
            'should_intervene': False,
        }


    if not api_key:
        return unavailable_assessment('LLM intervention assessment unavailable (missing OPENAI_API_KEY).')

    prompt = """Imagine you are the mediator in a buyer/seller purchase dispute. Let parties resolve it themselves unless intervention is necessary. Reasons:
1. Escalation of Conflict: if the conversation becomes heated with parties resorting to personal attacks or hostile language
2. Impasse: when parties reach a deadlock and are unable to move forward
3. Miscommunication: if there are signs that the parties are misunderstanding each other’s points
4. Unreasonable demands: If one party is making unreasonable demands that the other party can’t possibly meet.
5. Invocation: If one party asks the mediator to interject.

You will be given the conversation so far. Rate intervention urgency 1-5 (1=definitely don't intervene, 5=definitely intervene).
Provide JSON with: rating, reason (from list above), statement (one sentence).

You do not need to intervene every turn, and should consider how recently you've intervened before making a decision.

Return ENGLISH only."""

    payload = {
        "model": "gpt-4o",
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
        with urllib.request.urlopen(req, timeout=60) as resp:
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
        return unavailable_assessment('LLM intervention assessment unavailable (API request failed).')


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
        'reason': assessment.get('reason', 'Unavailable'),
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
        return _to_geo_payload('Unavailable', 0.0, {})
    try:
        if PREDICTOR is None:
            return _to_geo_payload('Unavailable', 0.0, {})
        outputs = PREDICTOR.predict_batch(role_turns)
        if not outputs:
            return _to_geo_payload('Unavailable', 0.0, {})

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
            pred, conf = 'Unavailable', 0.0
        return _to_geo_payload(pred, conf, averaged_probs)
    except Exception:
        return _to_geo_payload('Unavailable', 0.0, {})



def predict_cn_region_with_model(turns, role):
    role_turns = [t.get('text', '').strip() for t in turns if t.get('speaker') == role and t.get('text', '').strip()]
    base_province_probs = constrain_cn_probs_to_model_scope(infer_cn_province_distribution(turns, role))
    if not role_turns:
        return {
            'country': 'China',
            'confidence': 0.0,
            'probabilities': {'North': 0.0, 'Central': 0.0, 'Wu_Min': 0.0, 'Xian_Yue': 0.0},
            'region': 'Unavailable',
            'province_probabilities': {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()},
        }

    if PREDICTOR_ZH is None:
        return {
            'country': 'China',
            'confidence': 0.0,
            'probabilities': {'North': 0.0, 'Central': 0.0, 'Wu_Min': 0.0, 'Xian_Yue': 0.0},
            'region': 'Unavailable',
            'province_probabilities': {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()},
        }

    try:
        outputs = PREDICTOR_ZH.predict_batch(role_turns)
        combined = {}
        for out in outputs:
            for label, prob in (out.get('probabilities') or {}).items():
                combined[label] = combined.get(label, 0.0) + float(prob)
        count = len(outputs) or 1
        averaged = {k: v / count for k, v in combined.items()}
        total = sum(averaged.values()) or 1.0
        normalized = {k: round(v / total, 6) for k, v in averaged.items()}
        if not normalized:
            raise ValueError('empty region prediction')
        pred, conf = max(normalized.items(), key=lambda item: item[1])
        province_probs = project_region_probs_to_provinces(normalized, base_province_probs)
    except Exception:
        return {
            'country': 'China',
            'confidence': 0.0,
            'probabilities': {'North': 0.0, 'Central': 0.0, 'Wu_Min': 0.0, 'Xian_Yue': 0.0},
            'region': 'Unavailable',
            'province_probabilities': {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()},
        }
    return {
        'country': 'China',
        'confidence': round(conf, 2),
        'probabilities': normalized,
        'region': pred,
        'province_probabilities': province_probs,
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
