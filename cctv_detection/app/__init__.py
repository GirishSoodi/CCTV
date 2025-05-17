from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'cctv_detection_secret_key'
    
    from .routes import main
    app.register_blueprint(main)
    
    return app
