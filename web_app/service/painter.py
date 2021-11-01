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

def paint_s3_image(upload_sketch_access_key, upload_color_access_key, result_access_key):
    try:
        download_image(s3, './upload_sketch.png', upload_sketch_access_key)
        download_image(s3, './upload_color.png', upload_color_access_key)

        #region_detect_skimage('./upload.png')

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='reference_gan/config.yml', help='specifies config yaml file')
        params = parser.parse_args()

        if os.path.exists(params.config):
            config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
            solver = Solver(config, get_loader(config))
            print('test start')
            solver.test()
            print('test finish')
        else:
            print("Please check your config yaml file")
      

        image = cv2.imread('reference_gan/colorization_gan4/results/gan_image.jpg')
        #image = cv2.imread('danboo/gan_result.png')
        skeleton, region, flatten = segment(image)
        cv2.imwrite('./result.png', flatten)
        print('ok!')

        upload_image(s3, './result.png', result_access_key)
    except:
        return False
    return True