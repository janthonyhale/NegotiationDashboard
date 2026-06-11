import json


# ── KODIS Issue Definitions ───────────────────────────────────────────────────
KODIS_ISSUES = {
    'refund':          {'label': 'Refund',              'options': ['Full (100%)', 'Half (50%)', 'None (0%)'],  'values': [1.0, 0.5, 0.0]},
    'buyer_review':    {'label': 'Buyer Review Kept / Removed', 'options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
    'seller_review':   {'label': 'Seller Review Kept / Removed','options': ['Yes', 'No'],                              'values': [1.0, 0.0]},
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
        own_review_removed = outcome['buyer_review']
        other_review_removed = outcome['seller_review']
        own_apology = outcome['buyer_apology']
        other_apology = outcome['seller_apology']
        refund_component = outcome['refund']
        # Buyer gets points when own review stays up (remove=no) and when seller removes theirs (remove=yes).
        u  = weights['refund']         * refund_component
        u += weights['buyer_review']   * (1 - own_review_removed)
        u += weights['seller_review']  * other_review_removed
        u += weights['seller_apology'] * other_apology
        u += weights['buyer_apology']  * (1 - own_apology)
    else:  # seller
        own_review_removed = outcome['seller_review']
        other_review_removed = outcome['buyer_review']
        own_apology = outcome['seller_apology']
        other_apology = outcome['buyer_apology']
        refund_component = (1 - outcome['refund'])
        # Seller gets points when own review stays up (remove=no) and when buyer removes theirs (remove=yes).
        u  = weights['refund']         * refund_component
        u += weights['seller_review']  * (1 - own_review_removed)
        u += weights['buyer_review']   * other_review_removed
        u += weights['buyer_apology']  * other_apology
        u += weights['seller_apology'] * (1 - own_apology)
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
