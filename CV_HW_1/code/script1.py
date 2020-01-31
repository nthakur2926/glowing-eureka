from one_more_try_q1 import alignChannels
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load('red.npy')
green = np.load('green.npy')
blue = np.load('blue.npy')

rgbResult = alignChannels(red, green, blue)

plt.figure()
plt.imshow(rgbResult)

plt.show()
scipy.misc.imsave('../results/rgb_output.jpg', rgbResult)


