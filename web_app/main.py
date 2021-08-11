from flask import Flask
from app import views

app = Flask(__name__)

app.add_url_rule('/', 'index', views.index, methods=['GET', 'POST'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)