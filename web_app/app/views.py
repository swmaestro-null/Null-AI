from flask import render_template, request
from app.edge_detector import edge_detect
from app.s3_connector import s3_connection, upload_image

UPLOAD_FLODER = 'static/uploads/'
RESULT_FLODER = 'static/result/'
s3 = s3_connection()

def index():
    if request.method == "POST":
        f = request.files['image']
        filename = f.filename
        print(filename)
        f.save('./upload.png')
        upload_image(s3, './upload.png', UPLOAD_FLODER + filename)
        edge_detect()
        upload_image(s3, './result.png', RESULT_FLODER + filename)

        return render_template('index.html', fileupload=True, img_name=filename)

    return render_template('index.html', fileupload=False)