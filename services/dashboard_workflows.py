from services.dashboard_helpers import (
    CN_PROVINCE_CENTROIDS,
    NEG_SIGNALS,
    _irp_patterns,
    extract_uploaded_bundle_metadata,
    get_advisor,
    llm_detect_agreement_last_two,
    llm_emotion_scores,
    llm_executive_brief,
    llm_failure_risk,
    llm_irp_label,
    llm_operational_summary,
    llm_translate_cn_to_en,
    make_cn_province_map,
    make_emotion_plot,
    make_pareto_plot,
    predict_cn_region_with_model,
    predict_country_with_model,
    turns_have_cached_enrichment,
)
from services.kodis import (
    DEFAULT_BUYER_WEIGHTS,
    DEFAULT_SELLER_WEIGHTS,
    compute_pareto,
    extract_pre_dispute_justifications,
    extract_preference_weights,
    generate_all_outcomes,
)
from services.parsing import extract_dialogue_language, parse_file


def process_upload(path, ext, filename):
    turns = parse_file(path, ext)
    if not turns:
        raise ValueError('No turns found in file')

    bundle_meta = extract_uploaded_bundle_metadata(path, ext)
    language = bundle_meta.get('language', extract_dialogue_language(path, ext))
    has_cached = turns_have_cached_enrichment(turns)

    if language == 'CN':
        for t in turns:
            txt = t.get('text', '')
            if txt and not t.get('translation'):
                t['translation'] = llm_translate_cn_to_en(txt)
            trans = t.get('translation') or ''
            if trans:
                t['emotions'] = llm_emotion_scores(trans)
                if not t['emotions']:
                    t['emotion_error'] = 'LLM emotion classification unavailable.'
                tl = trans.lower()
                t['negative_signals'] = sum(1 for s in NEG_SIGNALS if s in tl)
                t['threat_signals'] = sum(1 for s in ['sue', 'lawyer', 'court', 'legal action'] if s in tl)

    pre_justifications = extract_pre_dispute_justifications(path, ext)
    if language == 'CN' and isinstance(pre_justifications, dict):
        for issue, vals in pre_justifications.items():
            if not isinstance(vals, dict):
                continue
            for side in ['buyer', 'seller']:
                txt = vals.get(side)
                if isinstance(txt, str) and txt.strip():
                    vals[side] = llm_translate_cn_to_en(txt.strip())
    bw, sw = extract_preference_weights(path, ext)
    calc_bw = {**DEFAULT_BUYER_WEIGHTS, **bw}
    calc_sw = {**DEFAULT_SELLER_WEIGHTS, **sw}
    outcomes = generate_all_outcomes(calc_bw, calc_sw)
    pareto   = compute_pareto(outcomes)
    pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: KODIS Solution Space')

    imported_country = bundle_meta.get('country') if isinstance(bundle_meta, dict) else None
    has_imported_country = (
        isinstance(imported_country, dict)
        and isinstance(imported_country.get('buyer'), dict)
        and isinstance(imported_country.get('seller'), dict)
    )
    if has_imported_country:
        buyer_c = imported_country.get('buyer', {})
        seller_c = imported_country.get('seller', {})
    elif language == 'CN':
        buyer_c = predict_cn_region_with_model(turns, 'Buyer')
        seller_c = predict_cn_region_with_model(turns, 'Seller')
        empty_probs = {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()}
        buyer_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='buyer')
        seller_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='seller')
    else:
        buyer_c  = predict_country_with_model(turns,'Buyer')
        seller_c = predict_country_with_model(turns,'Seller')

    if language == 'CN':
        empty_probs = {k: 0.0 for k in CN_PROVINCE_CENTROIDS.keys()}
        buyer_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='buyer')
        seller_c['province_map_b64_empty'] = make_cn_province_map(empty_probs, role='seller')

    return {
        'turns':turns,'filename':filename,
        'language': language,
        'buyer_weights':bw,'seller_weights':sw,
        'outcomes':outcomes,'pareto':pareto,
        'pareto_img':pareto_img,
        'country':{'buyer':buyer_c,'seller':seller_c},
        'pre_justifications': pre_justifications,
        'op_summaries': bundle_meta.get('op_summaries', []),
        'step_cache': bundle_meta.get('step_cache', {}),
    }


def update_weights_response(data):
    bw   = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw   = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    outcomes = generate_all_outcomes(bw, sw)
    pareto   = compute_pareto(outcomes)
    pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: KODIS Solution Space')
    return {'outcomes':outcomes,'pareto':pareto,'pareto_img':pareto_img}


def step_response(data):
    idx   = data.get('idx',0)
    turns = data.get('turns',[])
    dim   = data.get('emotion_dim','joy')
    language = str(data.get('language', 'EN')).upper()
    if not turns or idx >= len(turns):
        raise ValueError('Invalid')

    cur = turns[idx]
    emo_text = cur.get('text', '')
    if language == 'CN' and cur.get('text') and not cur.get('translation'):
        cur['translation'] = llm_translate_cn_to_en(cur.get('text',''))
    if language == 'CN':
        emo_text = cur.get('translation') or cur.get('text', '')
        cur['emotions'] = llm_emotion_scores(emo_text)
        if not cur['emotions']:
            cur['emotion_error'] = 'LLM emotion classification unavailable.'
    elif not cur.get('emotions'):
        cur['emotions'] = llm_emotion_scores(cur.get('text',''))
        if not cur['emotions']:
            cur['emotion_error'] = 'LLM emotion classification unavailable.'
    tl = str(emo_text or '').lower()
    cur['negative_signals'] = sum(1 for s in NEG_SIGNALS if s in tl)
    cur['threat_signals'] = sum(1 for s in ['sue', 'lawyer', 'court', 'legal action'] if s in tl)
    turns_so_far = turns[:idx+1]

    irp_label = llm_irp_label(turns_so_far, cur)
    cur.setdefault('meta', {})
    cur['meta']['irp_label'] = irp_label
    cur['irp'] = irp_label.lower()

    risk  = llm_failure_risk(turns_so_far)
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
    irp_patterns = _irp_patterns(turns_so_far)
    op_summary = llm_operational_summary(turns_so_far, country_snapshot, risk, irp_patterns)
    return {
        'risk':risk,
        'advisor':adv,
        'emotion_img':emo_img,
        'current_turn':cur,
        'country':country_snapshot,
        'operational_summary': op_summary,
    }


def post_summary_response(data):
    data = data or {}
    turns = data.get('turns', [])
    language = str(data.get('language', 'EN')).upper()
    bw    = data.get('buyer_weights', DEFAULT_BUYER_WEIGHTS)
    sw    = data.get('seller_weights', DEFAULT_SELLER_WEIGHTS)
    op_summaries = data.get('op_summaries', [])
    provided_final_outcome = data.get('final_outcome')

    turns_enriched = turns
    risk = llm_failure_risk(turns_enriched)
    outcomes = generate_all_outcomes(bw, sw)
    pareto = compute_pareto(outcomes)

    # Prefer LLM-based agreement detection using only the final two turns.
    final_outcome = llm_detect_agreement_last_two(turns_enriched) or provided_final_outcome

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
    return {
        'risk': risk,
        'post_img': post_img,
        'final_outcome': match,
        'outcomes': outcomes,
        'pareto': pareto,
        'executive_brief': executive_brief,
    }
