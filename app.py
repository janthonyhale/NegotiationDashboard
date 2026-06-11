import os

from flask import Flask

from routes.dashboard import dashboard_bp


app = Flask(__name__)
app.secret_key = 'nego_dash_kodis_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.register_blueprint(dashboard_bp)


if __name__=='__main__':
    os.makedirs('uploads',exist_ok=True)
    app.run(debug=True,port=5001)
