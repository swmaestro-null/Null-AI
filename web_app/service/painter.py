#from app.region_detector import region_detect_skimage
#from app.s3_connector import s3_connection, upload_image, download_image
from service.region_detector import region_detect_skimage
from service.s3_connector import s3_connection, upload_image, download_image

s3 = s3_connection()

def paint_s3_image(reference_access_key, sketch_access_key, result_access_key):
    try:
        # download image from s3
        download_image(s3, './reference.png', reference_access_key)
        download_image(s3, './sketch.png', sketch_access_key)
        # do paint
        region_detect_skimage('./sketch.png')
        # upload result image to s3
        resultUrl = upload_image(s3, './result.png', result_access_key)
        return {
            "resultUrl": resultUrl,
            "success": True
        }
    except Exception as e:
        print(e)
        return {
            "success": False
        }