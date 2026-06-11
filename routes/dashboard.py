import os

from flask import Blueprint, current_app, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from services.dashboard import (
    allowed,
    build_pdf_report,
    enriched_export_buffer,
    post_summary_response,
    process_upload,
    step_response,
    update_weights_response,
)


dashboard_bp = Blueprint('dashboard', __name__)


# ── Routes ────────────────────────────────────────────────────────────────────
@dashboard_bp.route('/')
def index(): return render_template('upload.html')

@dashboard_bp.route('/pre')
def pre(): return render_template('pre.html')

@dashboard_bp.route('/negotiate')
def negotiate(): return render_template('negotiate.html')

@dashboard_bp.route('/post')
def post(): return render_template('post.html')

@dashboard_bp.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files: return jsonify({'error':'No file'}),400
    f = request.files['file']
    if not f or not allowed(f.filename): return jsonify({'error':'Invalid file type'}),400
    filename = secure_filename(f.filename)
    ext = filename.rsplit('.',1)[1].lower()
    path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    try:
        return jsonify(process_upload(path, ext, filename))
    except ValueError as e:
        return jsonify({'error':str(e)}),400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error':str(e)}),500

@dashboard_bp.route('/api/update_weights', methods=['POST'])
def api_update_weights():
    return jsonify(update_weights_response(request.json))

@dashboard_bp.route('/api/step', methods=['POST'])
def api_step():
    try:
        return jsonify(step_response(request.json))
    except ValueError as e:
        return jsonify({'error':str(e)}),400

@dashboard_bp.route('/api/post_summary', methods=['POST'])
def api_post_summary():
    return jsonify(post_summary_response(request.json))

@dashboard_bp.route('/api/export_pdf', methods=['POST'])
def api_export_pdf():
    buf = build_pdf_report(request.json)
    return send_file(buf,mimetype='application/pdf',as_attachment=True,download_name='negotiation_report.pdf')

@dashboard_bp.route('/api/export_enriched', methods=['POST'])
def api_export_enriched():
    buf = enriched_export_buffer(request.json)
    return send_file(buf, mimetype='application/json', as_attachment=True, download_name='negotiation_enriched.json')
