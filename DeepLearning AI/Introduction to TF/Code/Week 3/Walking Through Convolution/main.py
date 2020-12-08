import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

i = misc.ascent()
row, cols = plt.subplots(1, 3)
plt.grid(False)
plt.gray()
cols[0].imshow(i)

i_copy = i.copy()
size_x = i_copy.shape[0]
size_y = i_copy.shape[1]

filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [[-1, -1, -2], [0, 0, 0], [1, 2, 1]] # Vertical lines
filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # Horizontal lines
weight = 1

#--------------------------------------------------------------------------#
# This part handles the Convolution
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0
        convolution = convolution + (i[x-1, y-1] * filter[0][0])
        convolution = convolution + (i[x, y-1] * filter[0][1])
        convolution = convolution + (i[x+1, y-1] * filter[0][2])
        convolution = convolution + (i[x-1, y] * filter[1][0])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x+1, y] * filter[1][2])
        convolution = convolution + (i[x-1, y+1] * filter[2][0])
        convolution = convolution + (i[x, y+1] * filter[2][1])
        convolution = convolution + (i[x+1, y+1] * filter[2][2])
        convolution *= weight
        if convolution < 0:
            convolution = 0
        if convolution > 255:
            convolution = 255
        i_copy[x][y] = convolution
#--------------------------------------------------------------------------#

cols[1].imshow(i_copy)

#--------------------------------------------------------------------------#
# This part handles the pooling of the image
new_x = int(size_x/2)
new_y = int(size_y/2)
new_image = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixel = []
        pixel.append(i_copy[x, y])
        pixel.append(i_copy[x+1, y])
        pixel.append(i_copy[x, y+1])
        pixel.append(i_copy[x+1, y+1])
        pixel.sort(reverse=True)
        new_image[int(x/2), int(y/2)] = pixel[0]
#--------------------------------------------------------------------------#

cols[2].imshow(new_image)

plt.show()


