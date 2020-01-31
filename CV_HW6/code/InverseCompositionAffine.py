import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

def InverseCompositionAffine(It, It1):
    
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).flatten()
    p_del = np.array([It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1]])
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dy, dx = np.gradient(It1)
    dx, dy = dx.reshape(-1), dy.reshape(-1)
    height = It.shape[0]
    width = It.shape[1]
    coor_x = np.arange(height)
    coor_y = np.arange(width)
    grid_x, grid_y = np.meshgrid(coor_x,coor_y)
    range_x1 = grid_x.shape[0]
    range_x2 = grid_x.shape[1]
    grid_x = np.reshape(grid_x,(1, np.multiply(range_x1,range_x2)))
    range_y1 = grid_y.shape[0]
    range_y2 = grid_y.shape[1]
    grid_y = np.reshape(grid_y,(1, np.multiply(range_y1,range_y2)))
    
    spline_It1 = RectBivariateSpline(coor_x, coor_y, It1)
    spline_It = RectBivariateSpline(coor_x, coor_y, It)
    deriv_y = spline_It.ev(grid_y, grid_x, dx=1, dy=0)
    deriv_y.flatten() 
    matrix_2 = np.zeros((It.shape[1]*It.shape[0])).astype(np.float32)
    deriv_x = spline_It.ev(grid_y, grid_x, dx=0, dy=1)
    deriv_x.flatten()
    matrix_1 = np.empty((np.multiply(It.shape[1],It.shape[0]), 6)).astype(np.float32)
    
    matrix_1[:, 0], matrix_1[:, 1],matrix_1[:, 2]  = np.multiply(dx,grid_x), np.multiply(dy,grid_y), dx
    matrix_1[:, 3], matrix_1[:, 4], matrix_1[:, 5] = np.multiply(dy,grid_x), np.multiply(dy,grid_y), grid_y
    dot_prod2 = np.dot(np.transpose(matrix_1), matrix_1)
    inverse = np.linalg.pinv(dot_prod2)

    p_delta = np.dot(inverse, np.transpose(matrix_1))
#    p_delta = p_delta.flatten()
    threshold =0.001
    p_delta_log = threshold
    mask = np.ones((It.shape[0], It.shape[1]))
    while p_delta_log >= 0.001:
        image_warped = cv2.warpAffine(It, M, (width, height))
		
        prod = np.multiply(It1, cv2.warpAffine(mask, M, (width, height)))
        matrix_2 = (prod - image_warped).reshape((np.multiply(It.shape[1],It.shape[0]), 1))
        p_delta_log = np.linalg.norm(np.dot(p_delta,matrix_2))
        p_del = np.dot(p_delta,matrix_2)
        a1 = 1.0 + p_del[0]
        a2 = p_del[1]
        a3 = p_del[2]
        b1 = p_del[3]
        b2 = 1.0 + p_del[4]
        b3 = p_del[5]
        array = np.array([[a1, a2, a3], 
                          [b1, b2, b3],
                          [0.0, 0.0, 1.0]])
        array = array.astype(np.float32)
        inverse2 = np.linalg.inv(array)
        M = np.dot(M, inverse2)
    return M


    