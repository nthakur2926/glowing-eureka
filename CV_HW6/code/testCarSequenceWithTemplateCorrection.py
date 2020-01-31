import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2
# write your script here, we recommend the above libraries for making your animation
from LucasKanade import LucasKanade

p0 = None
rectangle_list = np.empty((0, 4))
dataset = np.load('../data/carseq.npy')
rectangle = np.array([59, 116, 145, 151])
reference_frame = dataset[:, :, 0]
frame_num = dataset.shape[2]
rectangle_list = np.append(rectangle_list, rectangle)

#width = rect[2] - rect[0]
#height = rect[3] - rect[1]
rectListOriginal = np.load('carseqrects.npy')
sigma = 0.5

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("Template Correction")

print("Lucas Kanade tracking...")
for item in range(1, frame_num):
    print("num_frame", item)
    height = dataset[:, :, item].shape[0]
    width = dataset[:, :, item].shape[1]
    image = np.zeros((height, width, 3))
    p = LucasKanade(reference_frame, dataset[:, :, item], rectangle)
    
    
    coor_1 = rectangle[0]+p[0]
    coor_2 = rectangle[1]+p[1]
    coor_3 = rectangle[2]+p[0]
    coor_4 = rectangle[3]+p[1]
    updated_rectangle = [coor_1, coor_2, coor_3, coor_4]
    ref = dataset[:, :, item]
    rectangle_list = np.append(rectangle_list, updated_rectangle)
    
    image[:, :, 0], image[:, :, 1], image[:, :, 2] = ref, ref, ref
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    rect1 = patches.Rectangle((rectangle[0], rectangle[1]), rectangle[2], rectangle[3], linewidth=2, edgecolor='red', fill=False)
#    ax.add_patch(rect1)
#    plt.imshow(image, cmap='gray')
#    rect1 = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
    x1 = round(updated_rectangle[0])
    y1 = round(updated_rectangle[1])
    x2 = round(updated_rectangle[2])
    y2 = round(updated_rectangle[3])
    point1 = (int(x1),int(y1))
    point2 = (int(x2), int(y2))
    
    cv2.rectangle(image, point1, point2, color=(0,0,255))
    
#    cv2.putText(image, "Tracked_Car", 
#                        int(int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1, (255,255,255), 3)
    x_1 = round(rectListOriginal[item][0])
    y_1 = round(rectListOriginal[item][1])
    x_2 = round(rectListOriginal[item][2])
    y_2 = round(rectListOriginal[item][3])
    cv2.rectangle(image, (int(x_1), int(y_1)),
                      (int(x_2), int(y_2)),
                      color=(255,0,0))
    plt.title("Lucas Kanade Tracking with template correction")
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF
#    sub = p-p0
#    v = np.sum(np.multiply(sub,sub))
#    squareroot = np.sqrt(v)
    if p0 is None:
        reference_frame = dataset[:, :, item]
        rectangle = updated_rectangle
        p0 = p
    
    elif np.sqrt(np.sum(np.multiply((p-p0),(p-p0)))) < sigma:
        reference_frame = dataset[:, :, item]
        rectangle = updated_rectangle
        p0 = p
    else:
        reference_frame = reference_frame
        rectangle = rectangle
        p0 = p0
        

rectangle_list = np.append(rectangle_list, rectangle.reshape((1,4)), axis=0)
np.save('carseqrects-wcrt.npy', rectangle_list)
array = [1, 100, 200, 300, 400]
for item in range(1, frame_num):
    if item in array:
         cv2.imwrite("../code/", 'corrected_{}.jpg'.format(item), image)
        
    