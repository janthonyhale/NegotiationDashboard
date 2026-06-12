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


DEFAULT_ISSUE_DEFS = [
    {'key': 'refund', 'label': 'Refund', 'options': [
        {'label': 'Full', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 1.0},
        {'label': 'Half', 'buyer_value': 0.5, 'seller_value': 0.5, 'value': 0.5},
        {'label': 'None', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 0.0},
    ]},
    {'key': 'buyer_review', 'label': 'Buyer Review Removed', 'options': [
        {'label': 'Yes', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 1},
        {'label': 'No', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 0},
    ]},
    {'key': 'seller_review', 'label': 'Seller Review Removed', 'options': [
        {'label': 'Yes', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 1},
        {'label': 'No', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 0},
    ]},
    {'key': 'seller_apology', 'label': 'Seller Apologizes', 'options': [
        {'label': 'Yes', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 1},
        {'label': 'No', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 0},
    ]},
    {'key': 'buyer_apology', 'label': 'Buyer Apologizes', 'options': [
        {'label': 'Yes', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 1},
        {'label': 'No', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 0},
    ]},
]


def default_issue_definitions():
    return json.loads(json.dumps(DEFAULT_ISSUE_DEFS))


def normalize_issue_definitions(raw):
    if not raw:
        return default_issue_definitions()
    items = raw.values() if isinstance(raw, dict) else raw
    if not isinstance(items, list) and not hasattr(items, '__iter__'):
        return default_issue_definitions()
    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        key = str(item.get('key') or item.get('id') or item.get('name') or f'issue_{idx+1}').strip().lower().replace(' ', '_')
        label = str(item.get('label') or item.get('name') or key.replace('_', ' ').title()).strip()
        options = item.get('options') or item.get('outcomes') or []
        norm_opts = []
        if isinstance(options, list):
            for j, opt in enumerate(options):
                if isinstance(opt, dict):
                    opt_label = str(opt.get('label') or opt.get('name') or opt.get('value') or f'Option {j+1}')
                    try:
                        bv = float(opt.get('buyer_value', opt.get('role1_value', opt.get('value_for_role1', opt.get('buyer', 1 if j == 0 else 0)))))
                    except Exception:
                        bv = 1.0 if j == 0 else 0.0
                    try:
                        sv = float(opt.get('seller_value', opt.get('role2_value', opt.get('value_for_role2', opt.get('seller', 1.0 - bv)))))
                    except Exception:
                        sv = 1.0 - bv
                    value = opt.get('value', j)
                else:
                    opt_label = str(opt)
                    bv = 1.0 if j == 0 else 0.0
                    sv = 1.0 - bv
                    value = j
                norm_opts.append({'label': opt_label, 'buyer_value': max(0.0, min(1.0, bv)), 'seller_value': max(0.0, min(1.0, sv)), 'value': value})
        if len(norm_opts) < 2:
            norm_opts = [
                {'label': 'Yes', 'buyer_value': 1.0, 'seller_value': 0.0, 'value': 1},
                {'label': 'No', 'buyer_value': 0.0, 'seller_value': 1.0, 'value': 0},
            ]
        out.append({'key': key, 'label': label, 'options': norm_opts})
    return out or default_issue_definitions()


def extract_issue_definitions(path, ext):
    try:
        data = {}
        if ext == 'json':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                data = json.load(f)
        elif ext == 'jsonl':
            with open(path, encoding='utf-8-sig', errors='replace') as f:
                first = f.readline().strip()
            data = json.loads(first) if first else {}
        if isinstance(data, dict):
            raw = data.get('issues') or data.get('negotiation_issues') or data.get('issue_definitions')
            if raw:
                return normalize_issue_definitions(raw)
    except Exception:
        pass
    return default_issue_definitions()


def has_complete_preferences(bw, sw, issues=None):
    keys = [i['key'] for i in (issues or default_issue_definitions())]
    return all(k in (bw or {}) for k in keys) and all(k in (sw or {}) for k in keys)


def normalize_weights_to_100(weights, issues=None):
    keys = [i['key'] for i in (issues or default_issue_definitions())]
    vals = {}
    for k in keys:
        try:
            vals[k] = max(0.0, float((weights or {}).get(k, 0)))
        except Exception:
            vals[k] = 0.0
    total = sum(vals.values())
    if total <= 0:
        return vals
    return {k: round(v / total * 100, 3) for k, v in vals.items()}

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

def generate_all_outcomes(bw, sw, issues=None):
    """Generate outcomes for configured issues with buyer/seller utility."""
    issues = normalize_issue_definitions(issues)
    bw = normalize_weights_to_100(bw, issues)
    sw = normalize_weights_to_100(sw, issues)
    outcomes = []

    def rec(i, selected):
        if i >= len(issues):
            total_b = sum(bw.values()) or 1
            total_s = sum(sw.values()) or 1
            bu = sum(float(bw.get(k, 0)) * float(v.get('buyer_value', 0)) for k, v in selected.items()) / total_b * 100
            su = sum(float(sw.get(k, 0)) * float(v.get('seller_value', 0)) for k, v in selected.items()) / total_s * 100
            outcome = {
                'issue_values': {k: v.get('value') for k, v in selected.items()},
                'issue_labels': {k: v.get('label') for k, v in selected.items()},
                'deal_label': '; '.join(f"{next((iss['label'] for iss in issues if iss['key'] == k), k)}: {v.get('label')}" for k, v in selected.items()),
                'buyer_util': round(bu, 1),
                'seller_util': round(su, 1),
            }
            # Preserve legacy fields for KODIS-specific outcome detection/UI.
            for k, v in selected.items():
                outcome[k] = v.get('value')
                if k == 'refund':
                    outcome['refund_label'] = v.get('label')
            outcomes.append(outcome)
            return
        issue = issues[i]
        for opt in issue.get('options', []):
            rec(i + 1, {**selected, issue['key']: opt})

    rec(0, {})
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
        try:
            out[key] = max(0, min(100, float(v)))
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
