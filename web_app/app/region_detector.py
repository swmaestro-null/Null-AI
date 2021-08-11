from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
matplotlib.use('Agg')

def region_detect_skimage(filename):
    pic = io.imread(fname=filename, as_gray=True)

    denoised = rank.median(pic, disk(2))
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(denoised, disk(2))
    labels = watershed(gradient, markers)

    plt.figure(figsize=(pic.shape[1]/300, pic.shape[0]/300), dpi=300)
    plt.imshow(pic)
    plt.imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
    plt.axis('off')
    plt.xticks([]), plt.yticks([])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig('./result.png', bbox_inches='tight', pad_inches=0)
