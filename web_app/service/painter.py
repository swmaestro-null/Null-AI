from app.region_detector import region_detect_skimage
from app.s3_connector import s3_connection, upload_image, download_image

s3 = s3_connection()

def paint_s3_image(upload_access_key, result_access_key):
    try:
        download_image(s3, './upload.png', upload_access_key)
        region_detect_skimage('./upload.png')
        upload_image(s3, './result.png', result_access_key)
    except:
        return False
    return True