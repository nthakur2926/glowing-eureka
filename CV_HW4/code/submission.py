"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import sympy as sp
import scipy
import cv2
from scipy.ndimage.filters import gaussian_filter
from math import *
from helper import *
from helper import refineF

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    
    image1 = cv2.imread('../data/im1.png')
    image2 = cv2.imread('../data/im2.png')
    N = pts1.shape[0]
    T = np.asarray([[1.0/M, 0.0, 0.0],
                  [0.0, 1.0/M, 0.0],
                  [0.0, 0.0, 1.0]])
    mat = np.zeros((N, 9))
    pts1 = pts1/M
    pts2 = pts2/M

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    mat[:, 0] = x2 * x1
    mat[:, 1] = x2 * y1
    mat[:, 2] = x2
    mat[:, 3] = y2 * x1
    mat[:, 4] = y2 * y1
    mat[:, 5] = y2
    mat[:, 6] = x1
    mat[:, 7] = y1
    mat[:, 8] = np.ones(N)
    
    p, q, v = np.linalg.svd(mat)
    F = np.transpose(v)[:,-1]
    F = np.transpose(F)
    F = np.reshape(F,[3,3])
    p_new, q_new, v_new = np.linalg.svd(F)
    a0 = q_new[0]
    a1 = q_new[1]
    a2 = 0
    q = np.array([[a0, 0.0, 0.0],
                  [0.0, a1, 0.0],
                  [0.0, 0.0, a2]])
    
    fund_matrix =np.dot(p_new, q)
    F =  np.dot(fund_matrix,v_new)
    F = refineF(F, pts1, pts2)
	# Un-normalize
    T_hash = np.transpose(T)
    F = np.dot(np.dot(T_hash,F), T)
    return F

'''
#Q2.2: Seven Point Algorithm
#    Input:  pts1, Nx2 Matrix
#            pts2, Nx2 Matrix
#            M, a scalar parameter computed as max (imwidth, imheight)
#    Output: Farray, a list of estimated fundamental matrix.
#'''
def sevenpoint(pts1, pts2, M):
    image1 = cv2.imread('../data/im1.png')
    image2 = cv2.imread('../data/im2.png')
    N = pts1.shape[0]
    T = np.array([[1.0/M, 0.0, 0.0],
                  [0.0, 1.0/M, 0.0],
                  [0.0, 0.0, 1.0]])
    mat = np.zeros((N, 9))
    pts1 = pts1/M
    pts2 = pts2/M

    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]

    mat[:, 0] = x2 * x1
    mat[:, 1] = x2 * y1
    mat[:, 2] = x2
    mat[:, 3] = y2 * x1
    mat[:, 4] = y2 * y1
    mat[:, 5] = y2
    mat[:, 6] = x1
    mat[:, 7] = y1
    mat[:, 8] = np.ones(N)
    
    p, q, v = np.linalg.svd(mat)
    f0 = np.transpose(v)[:,-1]
    f0 = np.reshape(f0,[3,3])
    f1 = np.transpose(v)[:,-2]
    f1 = np.reshape(f1,[3,3])
    
#    k = sp.symbols('k')
#    print("k = ", k)
#    expr = np.linalg.det(k * f0 + (1 - k) * f1)
#    print("expr = ", expr)
#    coeff = sp.lambdify(k,expr,modules=['math'])
#    print("coeff = ",coeff)
    coeff = lambda k: np.linalg.det(k * f0 + (1 - k) * f1)
    alpha = float(2)
    beta = float(1/2)
    b = coeff(1) - coeff(-1)
    c = coeff(2) - coeff(-2)
    s0 = coeff(0)
    s1 = alpha * (b)/ 3 - (c)/12
    s2 = beta * coeff(1) + beta * coeff(-1) - coeff(0)
    s_val = s0 - s1 - s2
    s3 = coeff(1) - s_val
    arr = np.array([s3, s2, s1, s0])
#    base = np.polynomial.polynomial.polyroots(arr)
    base = np.roots(arr)
    
    F_final_array = []
    for item in base:
        i = float(np.real(item))
        fun = f0 * i + f1 * (1 - i)
        p, q, v = np.linalg.svd(fun)
        q_new = np.asarray([[q[0], 0.0, 0.0],
                          [0.0, q[1], 0.0],
                          [0.0, 0.0, q[2]]])
        fund = np.dot(np.dot(p, q_new), v)
        T_hash = np.transpose(T)
        phen = np.dot(T_hash, fund)
        F = np.dot(phen,T)
        F_final_array.append(F)
    return F_final_array
    

'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    K2_hash = np.transpose(K2)
    z = np.dot(K2_hash,F)
    d = np.dot(z,K1)    
    return d

'''
#Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
#    Input:  C1, the 3x4 camera matrix
#            pts1, the Nx2 matrix with the 2D image coordinates per row
#            C2, the 3x4 camera matrix
#            pts2, the Nx2 matrix with the 2D image coordinates per row
#    Output: P, the Nx3 matrix with the corresponding 3D points per row
#            err, the reprojection error.
#'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    mat = np.zeros((4,4))
    m = np.ones((N, 1))
    B = np.zeros((N,4))
    x1 = pts1[:,0]
    y1 = pts1[:,1]
    x2 = pts2[:,0]
    y2 = pts2[:,1]
    for i in range(N):
        mat[0,:] = x1[i] * C1[2,:] - C1[0,:]
        mat[1,:] = y1[i] * C1[2,:] - C1[1,:]
        mat[2,:] = x2[i] * C2[2,:] - C2[0,:]
        mat[3,:] = y2[i] * C2[2,:] - C2[1,:]
       
        p, q, v = np.linalg.svd(mat)
        B[i, :] = (v.T[:,-1] / v.T[:,-1][3])

    homo_matrix = B
    est_pts1 = homo_matrix @ C1.T
    est_pts2 = homo_matrix @ C2.T
    
    norm_pts1 = est_pts1 /  est_pts1[:, -1].reshape((-1,1))
    norm_pts2 = est_pts2 /  est_pts2[:, -1].reshape((-1,1))
    
    norm_pts1 = norm_pts1[:,:-1]
    norm_pts2 = norm_pts2[:,:-1]

    error = np.sum((np.linalg.norm(norm_pts1 - pts1)**2) + (np.linalg.norm(norm_pts2- pts2)**2))

    return B[:,:-1],error
'''
#Q4.1: 3D visualization of the temple images.
#    Input:  im1, the first image
#            im2, the second image
#            F, the fundamental matrix
#            x1, x-coordinates of a pixel on im1
#            y1, y-coordinates of a pixel on im1
#    Output: x2, x-coordinates of the pixel on im2
#            y2, y-coordinates of the pixel on im2
#
#'''
def epipolarCorrespondence(im1, im2, F, x1, y1):

#    if im1.shape[2] == 3:
#        im1 = cv2.normalize(im1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#    elif im2.shape[2] == 3:
#        im2 = cv2.normalize(im2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    check_range = 10
    window_size = 5
    height = im2.shape[0]
    width = im2.shape[1]
    temp = np.Inf
    point1 = np.array([[x1], [y1], [1]])
    epipolarLine= np.dot(F, point1)
    epipolarLine = epipolarLine/ np.linalg.norm(epipolarLine)
    e = epipolarLine[0]
    f = epipolarLine[1]
    g = epipolarLine[2]

    patch_1 = im1[int(y1 - window_size + 1) : int(y1 + window_size), int(x1 - window_size +1) : int(x1 + window_size), 0]
    patch_1 = gaussian_filter(patch_1, sigma=1)
    x2 = 0
    y2 = 0
    for i in range(int(y1-window_size*check_range), int(y1+ window_size*check_range)):
        curr_x2 = (-f * i - g) / e
        if (curr_x2 -window_size) > 0 and curr_x2 +window_size < width and i-window_size > 0 and i+window_size <= height:
            patch_2 = im2[int(i-window_size +1):int(i+window_size),int(curr_x2-window_size + 1): int(curr_x2 + window_size), 0]
            patch_2 = gaussian_filter(patch_2, sigma=1)
            error = np.linalg.norm(patch_1 - patch_2)
# 
            if error < temp:
                temp = error
                x2 = curr_x2
                y2 = i
                
    return x2, y2 

#
#
#   
#    
#    