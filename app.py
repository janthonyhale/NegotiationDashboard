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
    'seller_apology':  {'label': 'Seller Apologizes',   'options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
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
    turns = []
    if ext == 'jsonl':
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                turns.append({'idx':i,'speaker':obj.get('speaker','Unknown'),'text':obj.get('text',''),'ts':None,'meta':{}})
    elif ext == 'json':
        with open(path) as f: data = json.load(f)
        items = data if isinstance(data,list) else data.get('turns', data.get('messages',[data]))
        for i,obj in enumerate(items):
            turns.append({'idx':i,'speaker':obj.get('speaker',obj.get('role','Unknown')),
                          'text':obj.get('text',obj.get('content','')),'ts':None,'meta':{}})
    elif ext == 'txt':
        with open(path) as f: content = f.read()
        pat = re.compile(r'^(Buyer|Seller|Mediator)\s*:\s*(.+)', re.M|re.I)
        for i,m in enumerate(pat.finditer(content)):
            turns.append({'idx':i,'speaker':m.group(1).capitalize(),'text':m.group(2).strip(),'ts':None,'meta':{}})
        if not turns:
            for i,line in enumerate(content.strip().split('\n')):
                line=line.strip()
                if not line: continue
                turns.append({'idx':i,'speaker':'Buyer' if i%2==0 else 'Seller','text':line,'ts':None,'meta':{}})
    elif ext == 'csv':
        with open(path,newline='') as f:
            reader = csv.DictReader(f)
            for i,row in enumerate(reader):
                speaker = row.get('speaker',row.get('Speaker',row.get('role','Unknown')))
                text    = row.get('text',row.get('Text',row.get('content','')))
                turns.append({'idx':i,'speaker':speaker,'text':text,'ts':None,'meta':{}})
    return turns

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
    neg = s['anger'] + s['fear']
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



def llm_emotion_scores(text):
    """Use GPT-4o for emotion recognition when OPENAI_API_KEY is available.
    Falls back to keyword heuristic on error/missing key.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return score_emo(text)

    prompt = (
        "Return ONLY JSON with keys anger,fear,joy,surprise,compassion,neutral,valence. "
        "All except valence in [0,1], valence in [-1,1]. Text: " + (text or "")
    )
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role":"system","content":"You are an emotion classifier for negotiation transcripts."},
            {"role":"user","content": prompt}
        ],
        "temperature": 0,
        "response_format": {"type":"json_object"}
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
        out = {
            'anger': float(obj.get('anger',0)),
            'fear': float(obj.get('fear',0)),
            'joy': float(obj.get('joy',0)),
            'surprise': float(obj.get('surprise',0)),
            'compassion': float(obj.get('compassion',0)),
            'neutral': float(obj.get('neutral',0)),
            'valence': float(obj.get('valence',0)),
        }
        for k in ['anger','fear','joy','surprise','compassion','neutral']:
            out[k] = max(0.0, min(1.0, out[k]))
        out['valence'] = max(-1.0, min(1.0, out['valence']))
        return out
    except Exception:
        return score_emo(text)

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

def get_advisor(turns_so_far, current_turn):
    risk    = estimate_risk(turns_so_far)
    anger   = current_turn['emotions'].get('anger', 0)
    fear    = current_turn['emotions'].get('fear', 0)
    should  = risk['score'] > 0.35 or anger > 0.4 or fear > 0.4
    success = round(max(0.05, 1 - risk['score'] - anger*0.3), 2)
    walkaway= round(min(0.95, risk['score'] + anger*0.2), 2)
    reasons = []
    if anger > 0.3:               reasons.append('Elevated anger detected')
    if fear  > 0.3:               reasons.append('Anxiety signals present')
    if risk['threats'] > 0:       reasons.append('Legal threat language used')
    if risk['negative_signals']>1:reasons.append('Multiple rejection signals')
    SUGG = {
        'Elevated anger detected':    'Acknowledge frustration; name the emotion before the issue.',
        'Anxiety signals present':    'Slow down; clarify each party\'s core concern explicitly.',
        'Legal threat language used': 'Redirect: "Let\'s see if we can resolve this here first."',
        'Multiple rejection signals': 'Propose a structured break; then introduce a new option.',
    }
    suggestion = SUGG.get(reasons[0], 'Progress is constructive. Encourage continued dialogue.') if reasons else 'No intervention needed. Talks are progressing well.'
    return {'should_intervene':should,'action':'Intervene' if should else 'Observe',
            'success_odds':success,'walkaway_odds':walkaway,'reasons':reasons,'suggestion':suggestion}

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


PREDICTOR = RegionPredictor(language='english', model_name='svm')

COUNTRY_LABEL_MAP = {
    'U.S.':'United States','US':'United States','United States':'United States',
    'U.K.':'United Kingdom','UK':'United Kingdom','United Kingdom':'United Kingdom',
    'Mexico':'Mexico','South Africa':'South Africa','Germany':'Germany','China':'China',
    'Japan':'Japan','India':'India','France':'France','Australia':'Australia','Canada':'Canada','Brazil':'Brazil'
}

def _to_geo_payload(label, confidence, probabilities=None):
    canonical = COUNTRY_LABEL_MAP.get(label, label)
    info = COUNTRY_SVG.get(canonical, {'lat':0,'lng':0,'flag':'🌍'})
    return {
        'country': canonical,
        'confidence': round(float(confidence or 0),2),
        'lat': info['lat'],
        'lng': info['lng'],
        'flag': info['flag'],
        'probabilities': probabilities or {},
    }

def predict_country_with_model(turns, role):
    text = ' '.join(t.get('text','') for t in turns if t.get('speaker') == role).strip()
    if not text:
        return _to_geo_payload('United States', 0.0, {})
    try:
        out = PREDICTOR.predict_one(text)
        pred = out.get('predicted_class','United States')
        conf = out.get('confidence',0)
        probs = out.get('probabilities',{})
        return _to_geo_payload(pred, conf, probs)
    except Exception:
        return predict_country(turns, role)

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
    return {'country':country,'confidence':round(conf,2),'lat':info['lat'],'lng':info['lng'],'flag':info['flag']}

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
        turns = enrich(turns)
        # Default KODIS weights
        bw = DEFAULT_BUYER_WEIGHTS
        sw = DEFAULT_SELLER_WEIGHTS
        outcomes = generate_all_outcomes(bw, sw)
        pareto   = compute_pareto(outcomes)
        pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: KODIS Solution Space')
        buyer_c  = predict_country_with_model(turns,'Buyer')
        seller_c = predict_country_with_model(turns,'Seller')
        return jsonify({
            'turns':turns,'filename':filename,
            'buyer_weights':bw,'seller_weights':sw,
            'outcomes':outcomes,'pareto':pareto,
            'pareto_img':pareto_img,
            'country':{'buyer':buyer_c,'seller':seller_c},
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
    if not turns or idx >= len(turns): return jsonify({'error':'Invalid'}),400

    cur = turns[idx]
    cur['emotions'] = llm_emotion_scores(cur.get('text',''))
    turns_so_far = turns[:idx+1]

    risk  = estimate_risk(turns_so_far)
    adv   = get_advisor(turns_so_far, cur)
    emo_img = make_emotion_plot(turns_so_far, dim)
    buyer_c = predict_country_with_model(turns_so_far, 'Buyer')
    seller_c = predict_country_with_model(turns_so_far, 'Seller')
    return jsonify({'risk':risk,'advisor':adv,'emotion_img':emo_img,'current_turn':cur,'country':{'buyer':buyer_c,'seller':seller_c}})

@app.route('/api/post_summary', methods=['POST'])
def api_post_summary():
    data = request.json or {}
    turns = data.get('turns', [])
    bw    = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw    = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    dim   = data.get('emotion_dim', 'joy')
    final_outcome = data.get('final_outcome')

    turns_enriched = enrich(turns) if turns and 'emotions' not in turns[0] else turns
    risk = estimate_risk(turns_enriched)
    emo_img = make_emotion_plot(turns_enriched, dim) if turns_enriched else ''
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
    return jsonify({
        'risk': risk,
        'emotion_img': emo_img,
        'post_img': post_img,
        'final_outcome': match,
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
    turns_enriched = enrich(turns) if turns and 'emotions' not in turns[0] else turns
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

if __name__=='__main__':
    os.makedirs('uploads',exist_ok=True)
    app.run(debug=True,port=5000)
