import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1):
    height = It.shape[0]
    width = It.shape[1]
    height1 = np.arange(It1.shape[0])
    width1 = np.arange(It1.shape[1])
   
    p_delta = np.array([It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1], It1.shape[1]])
    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])
    prior = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]]).flatten()
    mask = np.ones((It.shape[0], It.shape[1])).astype(np.float32)
    spline_It1 = RectBivariateSpline(height1, width1, It1)

    while np.sum(np.dot(p_delta,p_delta)) >= 0.001:
        grid_x, grid_y = np.meshgrid(np.arange(0, (It.shape[1]-1)+0.5), np.arange(0, (It.shape[0]-1)+0.5))
        a1 = np.multiply(prior[3],grid_x)
        a2 = np.multiply(prior[4],grid_y)
        range_y = a1 + a2 + prior[5]
        
        a3 = np.multiply(prior[0],grid_x)
        a4 = np.multiply(prior[1],grid_y)
        range_x = a3 + a4 + prior[2]
        
        warped_image = cv2.warpAffine(It, M, (It.shape[1], It.shape[0]))
        required = (range_x > 0) & (range_x < width) & (range_y > 0) & (range_y < height)   
        warped_image_masked = cv2.warpAffine(mask, M, (It.shape[1], It.shape[0]))
        image_masked = np.multiply(warped_image_masked,It1)
        grid_x, grid_y = grid_x[required], grid_y[required]
        f1 = It[required]
        f1.flatten()
        range_x, range_y = range_x[required], range_y[required]
        warped = spline_It1.ev(range_y, range_x).flatten()
        deriv_y = spline_It1.ev(range_y, range_x, dx=1, dy=0)
        deriv_y.flatten()
        deriv_x = spline_It1.ev(range_y, range_x, dx=0, dy=1)
        deriv_x.flatten()
        matrix_1 = np.zeros((deriv_x.shape[0], 6))
        matrix_2 = f1 - warped
        matrix = warped_image.flatten() - image_masked.flatten()
       
        matrix_1[:, 0], matrix_1[:, 1],matrix_1[:, 2]  = np.multiply(deriv_x,grid_x), deriv_y * grid_y, deriv_x
        matrix_1[:, 3], matrix_1[:, 4], matrix_1[:, 5] = deriv_y * grid_x, deriv_y * grid_y, grid_y

        dot_prod1 = np.dot(np.transpose(matrix_1), matrix_2)
        dot_prod2 = np.dot(np.transpose(matrix_1), matrix_1)
        inverse = np.linalg.pinv(dot_prod2)

        p_delta = np.dot(inverse, dot_prod1)
        p_delta = p_delta.flatten()

        prior = prior + p_delta
               
        M[0][0], M[0][1], M[0][2], M[1][0], M[1][1], M[1][2] = prior[0] + 1.0,prior[1], prior[2], prior[3],1.0 + prior[4],prior[5] 
                                                                   
    M = prior.reshape(2,3)
    return M
