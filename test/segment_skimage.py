# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_marked_watershed.html

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import io
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
import cv2

input_path = "sketch.PNG"
image = io.imread(input_path, cv2.IMREAD_COLOR)

# denoise image
denoised = rank.median(image, disk(2))

# find continuous region
markers = rank.gradient(denoised, disk(5)) < 10
markers = ndi.label(markers)[0]

# local gradient
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
ax[1].set_title("Local Gradient")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(image, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()