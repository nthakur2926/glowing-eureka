'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import *
from submission import *

file = np.load('../data/templeCoords.npz')
correspond = np.load('../data/some_corresp.npz')

image1 = cv2.imread('../data/im1.png')
image2 = cv2.imread('../data/im2.png')
height = image1.shape[0]
width = image1.shape[1]
M = max(height,width, image2.shape[0],image2.shape[1])

pts1 = correspond['pts1']
pts2 = correspond['pts2']

F = eightpoint(pts1, pts2, M)
print("F",F)
x1 = file['x1']
y1 = file['y1']

x2 = np.zeros((x1.shape[0], 1))
y2 = np.zeros((x1.shape[0], 1))

for item in range(x1.shape[0]):
    raw_x2, raw_y2  = epipolarCorrespondence(image1,image2, F, x1[item], y1[item])
    x2[item] = raw_x2
    y2[item] = raw_y2
p1 = np.append(x1, y1, axis=1)
p2 = np.append(x2, y2, axis=1)

intrinsics = np.load('../data/intrinsics.npz')

K1 = intrinsics['K1']
K2 = intrinsics['K2']

E = essentialMatrix( F, K1, K2 )
print("E = ", E)
M1 =np.asarray([[1.0,0.0,0.0,0.0],
      [0.0,1.0,0.0,0.0],
      [0.0,0.0,1.0,0.0]])
M2s = camera2(E)
C1 = np.dot(K1,M1)
prev_value = 0
for i in range(M2s.shape[2]):
    C2_temp = np.dot(K2, M2s[:,:,i])
    P_temp, error = triangulate(C1, p1, C2_temp, p2)
    if np.min(P_temp[:,2] < 0):
        continue

    if error > prev_value:
        P = P_temp
        C2 = C2_temp
        prev_value = error
        index = i

np.savez('q4_2.npz', M1=M1, M2=M2s[:,:,index], C1=C1, C2= C2)
print("find M2 error: ", error)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylabel('z')
ax.scatter(P[:, 0], P[:, 1], P[:, 2], cmap=plt.cm.coolwarm, marker='o')
plt.show()