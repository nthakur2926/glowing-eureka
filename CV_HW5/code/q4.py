import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage
import skimage.measure
import skimage.color as skc
import skimage.restoration as skr
import skimage.filters as skf
import skimage.morphology as skm
import skimage.segmentation

def findLetters(image):
    bboxes = []
    bw = None
    image = skr.denoise_bilateral(image, multichannel=True)
    
    gray_image = skc.rgb2gray(image)
    image_blur = skf.gaussian(gray_image, sigma=1.0)
    
    image_thres = skf.threshold_otsu(image_blur)
    binary = gray_image < image_thres
    bw = skm.closing(binary, skm.square(5))
    morph = skm.binary_opening(binary)

    image_labels = skm.label(morph, connectivity = 2)
    region_prop = skimage.measure.regionprops(image_labels)
    
    area = 0
    pad = 0
    for item in region_prop:
        area = area + item.area
    average_area = area / len(region_prop)

    for item in region_prop:
            if item.area >= average_area/ 2:
                x_, y_, xx_, yy_ = item.bbox
                point_x1, point_y1 = max(0, y_ - pad),max(0, x_ - pad)
                point_x2,point_y2 = min(image.shape[1], yy_ + pad),min(image.shape[0], xx_ + pad)
                bboxes.append(np.array([point_y1, point_x1, point_y2, point_x2]))

    bw = (1.0 - morph).astype(np.float)
    return bboxes, bw

