from flask import Flask, jsonify, request
from service import painter

app = Flask(__name__)

@app.route('/paint', methods=['POST'])
def paint():
    req = request.get_json()
    reference_access_key = req["referenceAccessKey"]
    sketch_access_key = req["sketchAccessKey"]
    result_access_key = req["resultAccessKey"]
    res = painter.paint_s3_image(reference_access_key, sketch_access_key, result_access_key)
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
