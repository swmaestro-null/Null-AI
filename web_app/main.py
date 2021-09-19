from flask import Flask, jsonify, request
from app.painter import paint_s3_image

app = Flask(__name__)

@app.route('/paint', methods=['POST'])
def paint():
    req = request.get_json()
    upload_access_key = req["uploadAccessKey"]
    result_access_key = req["resultAccessKey"]
    is_success = paint_s3_image(upload_access_key, result_access_key)
    if is_success:
        return jsonify({
            "message": "paint success",
            "success": True
        })
    else:
        return jsonify({
            "message": "paint fail",
            "success": False
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)