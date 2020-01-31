import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2

from SubtractDominantMotion import SubtractDominantMotion 

# write your script here, we recommend the above libraries for making your animation

dataset = np.load('../data/aerialseq.npy')
frame_num = dataset.shape[2]
reference_frame = dataset[:, :, 0]

fig = plt.figure()

for i in range(1, frame_num):
    mask = SubtractDominantMotion(reference_frame, dataset[:, :, i]) ==1
    index = mask
    height = dataset[:, :, i].shape[0]
    width = dataset[:, :, i].shape[1]
    image = np.zeros((height, width, 3))
    image = np.stack((dataset[:, :, i], dataset[:, :, i], dataset[:, :, i]), axis=2)
    image[:, :, 0][index] = 1
    cv2.imshow('image', image)
    cv2.waitKey(1)
    reference_frame = dataset[:, :, i]

    if i in [30, 60, 90, 120]:
        filename = 'AerialSequence.jpg'
        cv2.imwrite(filename.format(i), image)
    plt.pause(0.05)
    