import base64
import io
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.dashboard_helpers import (
    estimate_risk,
    llm_evolution_summary,
    make_all_emotions_plot,
    make_pareto_plot,
)
from services.kodis import DEFAULT_BUYER_WEIGHTS, DEFAULT_SELLER_WEIGHTS, compute_pareto, generate_all_outcomes


def build_pdf_report(data):
    data         = data or {}
    turns        = data.get('turns',[])
    language     = str(data.get('language', 'EN')).upper()
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
    turns_enriched = turns
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
    labels = ['Refund','Buyer Review Kept / Removed','Seller Review Kept / Removed','Seller Apologizes','Buyer Apologizes']
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
            ['Buyer Review Kept / Removed', 'Yes' if final_outcome.get('buyer_review') else 'No'],
            ['Seller Review Kept / Removed','Yes' if final_outcome.get('seller_review') else 'No'],
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
    return buf


def enriched_export_buffer(data):
    data = data or {}
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
    return buf
