import cv2, matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
def edge_detect():
    pic = cv2.imread('./upload.png')

    pic_preprocessed = cv2.cvtColor(cv2.GaussianBlur(pic, (7,7), 0), cv2.COLOR_BGR2GRAY)
    pic_edges = cv2.bitwise_not(cv2.Canny(pic_preprocessed, threshold1=20, threshold2=60))
    plt.imshow(cv2.cvtColor(pic_edges, cv2.COLOR_GRAY2RGB))
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.savefig('./result.png', dpi=300)