import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


bbox_padding_x = 20
bbox_padding_y = 18

def centerpoint(bbox):
        point_y1, point_x1, point_y2, point_x2 = bbox
        fraction = (point_y2 + point_y1) / 2
        return fraction

for img in os.listdir('./images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('./images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    sorted(bboxes, key=centerpoint)
    letters_length = len(bboxes)
    alphabets = np.zeros((0,1024))
    height = bw.shape[0]
    width = bw.shape[1]  
    row_index = np.zeros((letters_length, 1))
        
    sorted_bbox = []
    last_row, pres_row_index = 0,0
    
    for item in range(0, letters_length):
        temp = bboxes[item]
        point_y1, point_x1, point_y2, point_x2 = temp
        centerpoint_y, centerpoint_x = (point_y1 + point_y2) / 2, (point_x1 + point_x2) / 2
        
        if (centerpoint_y - last_row) >= (point_y2 - point_y1):
            pres_row_index = pres_row_index + 1
        last_row, row_index[item] = centerpoint_y, pres_row_index
        sorted_bbox.append((point_y1, point_x1, point_y2, point_x2, pres_row_index))

    sorted_bbox = sorted(sorted_bbox, key=lambda p: (p[-1], (p[1] + p[3] / 2)))

    for item in range(0,len(sorted_bbox)):
        temp = sorted_bbox[item]
        point_y1, point_x1, point_y2, point_x2, pres_row_index = temp
        centerpoint_y, centerpoint_x = (point_y1 + point_y2) / 2, (point_x1 + point_x2) / 2
        coor_x, coor_y = (point_x2 - point_x1) / 2, (point_y2 - point_y1) / 2
        
        w1 = point_y1 - 18
        w2 = point_x1 - 20
        w3 = point_y2 + 18
        w4 = point_x2 + 20
        point_x1,point_y1 = max(0, w2), max(0,w1)
        point_x2, point_y2 = min(width, point_x2 + 20), min(height, w3)

        alphabet = bw[point_y1:point_y2+1, point_x1:point_x2+1]
        alphabet = skimage.morphology.binary_erosion(alphabet)
        sq_alphabet = (skimage.transform.resize(alphabet, (32, 32)).T).reshape(-1)

        alphabets = np.append(alphabets, sq_alphabet.reshape((1, 1024)), axis=0)

    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    h1 = forward(alphabets, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    value = np.argmax(probs, axis=1)
    
    letter_value = letters[value]  
    line, line_index = "", 0

    for item in range(0,letter_value.shape[0]):
        if row_index[item] == line_index:
            line = line + letter_value[item]
        else:
            print(line)
            line = "" + letter_value[item]
            line_index = row_index[item]
    print(line)

