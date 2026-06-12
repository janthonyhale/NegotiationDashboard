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
    llm_negotiation_qa,
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
    extract_issue_definitions,
    extract_pre_dispute_justifications,
    extract_preference_weights,
    generate_all_outcomes,
    has_complete_preferences,
    normalize_weights_to_100,
)
from services.parsing import extract_dialogue_language, parse_file
from services.annotation_store import record_annotation_correction
from services.llm_client import openai_chat_text
import json
import os
import re


def _role_metadata(bundle_meta, turns):
    raw_roles = bundle_meta.get('role_names') if isinstance(bundle_meta, dict) else None
    if not isinstance(raw_roles, dict):
        raw_roles = {}
    role_names = {
        'role1': str(raw_roles.get('role1') or raw_roles.get('disputant1') or raw_roles.get('party1') or raw_roles.get('buyer') or 'Disputant 1'),
        'role2': str(raw_roles.get('role2') or raw_roles.get('disputant2') or raw_roles.get('party2') or raw_roles.get('seller') or 'Disputant 2'),
    }
    background = bundle_meta.get('task_background') if isinstance(bundle_meta, dict) else None
    if not isinstance(background, str) or not background.strip():
        background = 'NO BACKGROUND INFO'

    observed = []
    for t in turns or []:
        speaker = str(t.get('speaker', '') or '').strip()
        if speaker and speaker.lower() not in {'unknown', 'mediator'} and speaker not in observed:
            observed.append(speaker)
        if len(observed) >= 2:
            break
    speaker_to_role = {}
    if observed:
        speaker_to_role[observed[0]] = 'role1'
    if len(observed) > 1:
        speaker_to_role[observed[1]] = 'role2'
    for canonical, key in [('Buyer', 'role1'), ('Seller', 'role2')]:
        speaker_to_role.setdefault(canonical, key)
    return role_names, background, speaker_to_role


def _apply_role_metadata(turns, role_names, task_background, speaker_to_role):
    for t in turns or []:
        speaker = str(t.get('speaker', '') or '').strip()
        role_key = speaker_to_role.get(speaker)
        if not role_key:
            low = speaker.lower()
            if low == 'buyer':
                role_key = 'role1'
            elif low == 'seller':
                role_key = 'role2'
            elif low == 'mediator':
                role_key = 'mediator'
        t['role_key'] = role_key or t.get('role_key') or 'other'
        t['display_speaker'] = role_names.get(t['role_key'], speaker or 'Unknown') if t['role_key'] in role_names else (speaker or 'Unknown')
        t.setdefault('meta', {})
        t['meta']['role_key'] = t['role_key']
        t['meta']['display_speaker'] = t['display_speaker']
        t['meta']['task_background'] = task_background
    return turns

def process_upload(path, ext, filename):
    turns = parse_file(path, ext)
    if not turns:
        raise ValueError('No turns found in file')

    bundle_meta = extract_uploaded_bundle_metadata(path, ext)
    role_names, task_background, speaker_to_role = _role_metadata(bundle_meta, turns)
    turns = _apply_role_metadata(turns, role_names, task_background, speaker_to_role)
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
    issues = extract_issue_definitions(path, ext)
    bw, sw = extract_preference_weights(path, ext)
    preferences_complete = has_complete_preferences(bw, sw, issues)
    if preferences_complete:
        bw = normalize_weights_to_100(bw, issues)
        sw = normalize_weights_to_100(sw, issues)
        outcomes = generate_all_outcomes(bw, sw, issues)
        pareto = compute_pareto(outcomes)
        pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: Solution Space')
    else:
        outcomes = []
        pareto = []
        pareto_img = ''

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
        'issues': issues,
        'preferences_complete': preferences_complete,
        'outcomes':outcomes,'pareto':pareto,
        'pareto_img':pareto_img,
        'country':{'buyer':buyer_c,'seller':seller_c},
        'pre_justifications': pre_justifications,
        'role_names': role_names,
        'task_background': task_background,
        'op_summaries': bundle_meta.get('op_summaries', []),
        'step_cache': bundle_meta.get('step_cache', {}),
    }



def _heuristic_convert_text(raw_text):
    lines = [ln.strip() for ln in str(raw_text or '').splitlines() if ln.strip()]
    turns = []
    role_order = []
    pattern = re.compile(r'^([^:\-\[][^:\n]{0,60}?)(?:\s*\[[^\]]+\])?\s*[:\-]\s*(.+)$')
    for line in lines:
        m = pattern.match(line)
        if m:
            speaker = m.group(1).strip().strip('"')
            text = m.group(2).strip()
        else:
            speaker = f'Disputant {(len(turns) % 2) + 1}'
            text = line
        if speaker and speaker not in role_order and speaker.lower() != 'mediator':
            role_order.append(speaker)
        turns.append({'speaker': speaker, 'text': text})
    role_names = {
        'role1': role_order[0] if len(role_order) > 0 else 'Disputant 1',
        'role2': role_order[1] if len(role_order) > 1 else 'Disputant 2',
    }
    return {'language': 'EN', 'role_names': role_names, 'task_background': 'NO BACKGROUND INFO', 'turns': turns}


def convert_arbitrary_transcript_response(file_storage):
    if not file_storage:
        raise ValueError('No file')
    raw = file_storage.read().decode('utf-8-sig', errors='replace')
    if not raw.strip():
        raise ValueError('Uploaded file is empty')

    converted = None
    if os.getenv('OPENAI_API_KEY'):
        prompt = (
            'Convert this negotiation/dispute transcript into strict JSON only with fields: '
            'language (EN or CN), role_names {role1, role2}, task_background string, issues array, turns array, and optional buyer_weights/seller_weights. '
            'Each turn must have speaker and text. Preserve participant names as role_names when available. '
            'For issues, include negotiated issues only when explicit. For each issue include key, label, and options with label plus role1_value/role2_value from 0 to 1 when clear. '
            'Only include buyer_weights/seller_weights if preferences are explicitly stated in the input; do not infer or invent weights. '
            'If role names are missing use Disputant 1 and Disputant 2. If background is missing use NO BACKGROUND INFO. '
            'Do not invent dialogue.\n\nTranscript:\n' + raw[:24000]
        )
        try:
            content = openai_chat_text({
                'model': 'gpt-4o',
                'messages': [
                    {'role': 'system', 'content': 'You reformat messy dispute transcripts into app-ready JSON. Return JSON only.'},
                    {'role': 'user', 'content': prompt},
                ],
                'temperature': 0,
                'response_format': {'type': 'json_object'},
            }, timeout=60)
            converted = json.loads(content or '{}')
        except Exception:
            converted = None
    if not isinstance(converted, dict) or not isinstance(converted.get('turns'), list):
        converted = _heuristic_convert_text(raw)

    converted.setdefault('language', 'EN')
    converted.setdefault('role_names', {'role1': 'Disputant 1', 'role2': 'Disputant 2'})
    converted.setdefault('task_background', 'NO BACKGROUND INFO')
    turns = []
    for item in converted.get('turns') or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get('text') or item.get('content') or '').strip()
        if not text:
            continue
        turns.append({'speaker': str(item.get('speaker') or item.get('role') or 'Unknown').strip() or 'Unknown', 'text': text})
    if not turns:
        raise ValueError('No dialogue turns found after conversion')

    header = {
        'language': str(converted.get('language') or 'EN').upper() if str(converted.get('language') or 'EN').upper() in {'EN', 'CN'} else 'EN',
        'role_names': converted.get('role_names') if isinstance(converted.get('role_names'), dict) else {'role1': 'Disputant 1', 'role2': 'Disputant 2'},
        'task_background': converted.get('task_background') if isinstance(converted.get('task_background'), str) and converted.get('task_background').strip() else 'NO BACKGROUND INFO',
    }
    if converted.get('issues'):
        header['issues'] = converted.get('issues')
    if isinstance(converted.get('buyer_weights'), dict):
        header['buyer_weights'] = converted.get('buyer_weights')
    if isinstance(converted.get('seller_weights'), dict):
        header['seller_weights'] = converted.get('seller_weights')
    jsonl = '\n'.join([json.dumps(header, ensure_ascii=False)] + [json.dumps(t, ensure_ascii=False) for t in turns]) + '\n'
    return {'filename': 'converted_dispute.jsonl', 'jsonl': jsonl, 'turn_count': len(turns), 'role_names': header['role_names'], 'task_background': header['task_background']}

def update_weights_response(data):
    data = data or {}
    issues = data.get('issues') or extract_issue_definitions('', '')
    bw = data.get('buyer_weights') or {}
    sw = data.get('seller_weights') or {}
    if not has_complete_preferences(bw, sw, issues):
        return {'outcomes': [], 'pareto': [], 'pareto_img': '', 'preferences_complete': False}
    bw = normalize_weights_to_100(bw, issues)
    sw = normalize_weights_to_100(sw, issues)
    outcomes = generate_all_outcomes(bw, sw, issues)
    pareto = compute_pareto(outcomes)
    pareto_img = make_pareto_plot(outcomes, pareto, title='Pre-Negotiation: Solution Space')
    return {'outcomes': outcomes, 'pareto': pareto, 'pareto_img': pareto_img, 'buyer_weights': bw, 'seller_weights': sw, 'preferences_complete': True}


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

    cur.setdefault('meta', {})
    if cur.get('meta', {}).get('irp_human_corrected'):
        irp_label = cur.get('meta', {}).get('irp_label') or cur.get('irp') or 'Unavailable'
    else:
        irp_label = llm_irp_label(turns_so_far, cur)
        cur['meta']['irp_label'] = irp_label
    cur['irp'] = str(irp_label).lower()

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
    issues = data.get('issues') or extract_issue_definitions('', '')
    bw    = data.get('buyer_weights') or {}
    sw    = data.get('seller_weights') or {}
    op_summaries = data.get('op_summaries', [])
    provided_final_outcome = data.get('final_outcome')

    turns_enriched = turns
    risk = llm_failure_risk(turns_enriched)
    preferences_complete = has_complete_preferences(bw, sw, issues)
    if preferences_complete:
        bw = normalize_weights_to_100(bw, issues)
        sw = normalize_weights_to_100(sw, issues)
        outcomes = generate_all_outcomes(bw, sw, issues)
        pareto = compute_pareto(outcomes)
    else:
        outcomes = []
        pareto = []

    # Prefer LLM-based agreement detection using only the final two turns.
    final_outcome = llm_detect_agreement_last_two(turns_enriched) or provided_final_outcome

    match = None
    if final_outcome:
        match = next((o for o in outcomes if
                      o.get('refund_label') == final_outcome.get('refund_label') and
                      o.get('buyer_review') == final_outcome.get('buyer_review') and
                      o.get('seller_review') == final_outcome.get('seller_review') and
                      o.get('seller_apology') == final_outcome.get('seller_apology') and
                      o.get('buyer_apology') == final_outcome.get('buyer_apology')), None)

    post_img = make_pareto_plot(outcomes, pareto, match, title='Post-Negotiation: Solution Space') if preferences_complete else ''
    executive_brief = llm_executive_brief(
        turns_enriched, op_summaries, match, pareto,
        role_names=data.get('role_names'),
        task_background=data.get('task_background'),
    )
    return {
        'risk': risk,
        'post_img': post_img,
        'final_outcome': match,
        'outcomes': outcomes,
        'pareto': pareto,
        'preferences_complete': preferences_complete,
        'issues': issues,
        'executive_brief': executive_brief,
    }



def annotation_correction_response(data):
    data = data or {}
    turns = data.get('turns') or []
    idx = int(data.get('idx', -1))
    annotation_type = str(data.get('annotation_type', '')).strip().lower()
    if idx < 0 or idx >= len(turns):
        raise ValueError('Invalid turn index')
    if annotation_type not in {'irp', 'emotion'}:
        raise ValueError('Invalid annotation type')
    turn = turns[idx]
    old_value = data.get('old_value')
    new_value = data.get('new_value')
    record = record_annotation_correction(
        data.get('filename', ''), idx, turn.get('speaker', ''), turn.get('text', ''),
        annotation_type, old_value, new_value,
    )
    return {'ok': True, 'record': record}


def qa_response(data):
    data = data or {}
    return {
        'answer': llm_negotiation_qa(
            data.get('turns', []),
            data.get('question', ''),
            op_summaries=data.get('op_summaries', []),
            final_outcome=data.get('final_outcome'),
            language=str(data.get('language', 'EN')).upper(),
            role_names=data.get('role_names'),
            task_background=data.get('task_background'),
        )
    }
