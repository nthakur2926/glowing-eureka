import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
import cv2

from LucasKanade import LucasKanade

rectangle_list = np.empty((0, 4))
dataset = np.load('../data/carseq.npy')
rectangle = np.array([59, 116, 145, 151])
reference_frame = dataset[:, :, 0]
frame_num = dataset.shape[2]
rectangle_list = np.append(rectangle_list, rectangle)

#fig = plt.figure()
#ax = fig.add_subplot(111)

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
    rectangle = [coor_1, coor_2, coor_3, coor_4]
    ref = dataset[:, :, item]
    rectangle_list = np.append(rectangle_list, rectangle)
    
    image[:, :, 0], image[:, :, 1], image[:, :, 2] = ref, ref, ref
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    rect1 = patches.Rectangle((rectangle[0], rectangle[1]), rectangle[2], rectangle[3], linewidth=2, edgecolor='red', fill=False)
#    ax.add_patch(rect1)
#    plt.imshow(image, cmap='gray')
#    rect1 = rectangle[0], rectangle[1], rectangle[2], rectangle[3]
    x1 = round(rectangle[0])
    y1 = round(rectangle[1])
    x2 = round(rectangle[2])
    y2 = round(rectangle[3])
    point1 = (int(x1),int(y1))
    point2 = (int(x2), int(y2))
    cv2.rectangle(image, point1, point2, color=(0,0,255))
#    cv2.putText(image, "Tracked_Car", 
#                        int(int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1, (255,255,255), 3)

    plt.title("Lucas Kanade Tracking")
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF
    reference_frame = dataset[:, :, item]

rectangle_list = np.array(rectangle_list)
np.save('carseqrects.npy', rectangle_list)

for item in range(1, frame_num):
    coor_1 = rectangle[0]+p[0]
    coor_2 = rectangle[1]+p[1]
    coor_3 = rectangle[2]+p[0]
    coor_4 = rectangle[3]+p[1]
    rectangle = [coor_1, coor_2, coor_3, coor_4]
    ref = dataset[:, :, item]
    rectangle_list = np.append(rectangle_list, rectangle)
    

    image[:, :, 0], image[:, :, 1], image[:, :, 2] = ref, ref, ref
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    array = [1, 100, 200, 300, 400]
    if item in array:
#        plt.savefig("Tracked Car"+str(item)+".jpg")
        cv2.imwrite("../code/", 'Tracked_car_{}.jpg'.format(item), image)
    plt.pause(0.01)
    reference_frame = dataset[:, :, item]

rectangle_list = np.array(rectangle_list)
cv2.destroyAllWindows()
