from flask import render_template, request
import os
from app.edge_detector import edge_detect

UPLOAD_FLODER = 'static/uploads'

def index():
    if request.method == "POST":
        f = request.files['image']
        filename = f.filename
        print(filename)
        path = os.path.join(UPLOAD_FLODER, filename)
        f.save(path)
        edge_detect(filename)

        return render_template('index.html', fileupload=True, img_name=filename)

    return render_template('index.html', fileupload=False)