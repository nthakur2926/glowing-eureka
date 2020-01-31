'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import cv2
from submission import *
from helper import *

correspond = np.load('../data/some_corresp.npz')
p1 = correspond['pts1']
p2 = correspond['pts2']

image1 = cv2.imread('../data/im1.png')
image2 = cv2.imread('../data/im2.png')
height = image1.shape[0]
width = image1.shape[1]
M = max(height,width, image2.shape[0],image2.shape[1])

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

F = eightpoint( p1, p2, M )
print("Fundamental_matrix = ")
for items in F:
    print(items)
E = essentialMatrix( F, K1, K2 )
print("Essential_matrix = ")
for items in E:
    print(items)
M2s = camera2(E)

M1 = np.asarray([[1,0,0,0],
      [0,1,0,0],
      [0,0,1,0]])
C1 = np.dot(K1,M1)
cubes = np.zeros((M2s.shape[2],1))
posit = np.zeros((M2s.shape[2],1))
err = np.zeros((M2s.shape[2],1))
 
for i in range(4):
    C2 = np.dot(K2, M2s[:,:,i])
    P, error = triangulate( C1, p1, C2, p2 )
    if ((P < 0).sum() == 0):
        break

exact = np.argwhere(posit == P.shape[0])
M2 = M2s[:,:,exact]
print("M2 error = ", error)

np.savez('q3_3.npz', M2=M2, C2=C2)