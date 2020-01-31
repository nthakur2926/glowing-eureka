import numpy as np
import matplotlib.pyplot as plt
from submission import *
from helper import *
from mpl_toolkits.mplot3d import Axes3D



im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

correspond = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')
M = max(im1.shape)

K1 = intrinsics['K1']
K2 = intrinsics['K2']

pts1 = correspond['pts1']
pts2 = correspond['pts2']


''' Question 2 (2.1) '''
F = eightpoint(pts1, pts2, M)
np.savez('q2_1.npz', F=F, M=M)
displayEpipolarF(im1, im2, F)
print("Fundamental Matrix = ",F)


''' Question 2 (2.2) '''
m = 7
n = 2
rand = np.random.choice(pts1.shape[0], m)
point1 = np.zeros((m,n))
point2 = np.zeros((m, n))
for k in range(rand.shape[0]):
	point1[k] = pts1[rand[k]]
	point2[k] = pts2[rand[k]]

F0 = sevenpoint(point1, point2, M)
np.savez('q2_2.npz', F=F0, M=M, pts1=point1, pts2=point2)
for item in F0:
    print("F0_7point = ",item)
    displayEpipolarF(im1, im2, item)



''' Q4.1 '''
F = eightpoint(pts1, pts2, M)
epipolarMatchGUI(im1, im2, F)
np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)