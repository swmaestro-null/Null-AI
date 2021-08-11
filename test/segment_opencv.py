# https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html

from __future__ import print_function
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import random as rng

input_path = "sketch.PNG"
input_img = cv.imread(input_path, cv.IMREAD_COLOR)
src = cv.imread(input_path, cv.IMREAD_COLOR)
src[np.all(src == 255, axis=2)] = 0
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian

# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

# Normalize the distance image for range = {0.0, 1.0}
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
dist_8u = dist.astype('uint8')

# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create the marker image
markers = np.zeros(dist.shape, dtype=np.int32)

# Draw the foreground markers
for i in range(len(contours)):
    cv.drawContours(markers, contours, i, (i+1), -1)

# Draw the background marker
cv.circle(markers, (5,5), 3, (255,255,255), -1)
mark = (markers * 10).astype('uint8')


cv.watershed(imgResult, markers)

# Generate random colors
colors = []
for contour in contours:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))

# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours):
            dst[i,j,:] = colors[index-1]

# Visualize results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(input_img)
ax[0].set_title("Source Image")

ax[1].imshow(bw)
ax[1].set_title("Binary Image")

ax[2].imshow(mark)
ax[2].set_title("Markers")

ax[3].imshow(dst)
ax[3].set_title("Final Result")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()