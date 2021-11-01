import cv2
import os
import argparse
import yaml
import sys
sys.path.append("..")
from service.region_detector import region_detect_skimage
from service.s3_connector import s3_connection, upload_image, download_image
from reference_gan.solver import Solver 
from reference_gan.data_loader import get_loader
from danboo.segment import segment


s3 = s3_connection()


def paint_s3_image(reference_access_key, sketch_access_key, result_access_key):
    try:
        # download image from s3
        download_image(s3, './reference.png', reference_access_key)
        download_image(s3, './sketch.png', sketch_access_key)
        # do paint
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='reference_gan/config.yml', help='specifies config yaml file')
        params = parser.parse_args()

        if os.path.exists(params.config):
            config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
            solver = Solver(config, get_loader(config))
            print('test start')
            solver.test()
          
        else:
            print("Please check your config yaml file")
      
        image = cv2.imread('reference_gan/colorization_gan4/results/gan_image.jpg')
        #image = cv2.imread('danboo/gan_result.png')
        skeleton, region, flatten = segment(image)
        cv2.imwrite('./result.png', flatten)

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

