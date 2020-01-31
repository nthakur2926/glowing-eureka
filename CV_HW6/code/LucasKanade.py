import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    
    x = It.shape[0]
    y = It.shape[1]
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    threshold = 0.005
    p_del = np.array([y,x])

#    print("delp", p_del)
    coor_x1 = np.arange(It1.shape[0])
    coor_y1 = np.arange(It1.shape[1])
    spline_It1 = RectBivariateSpline(coor_x1, coor_y1, It1)
#    print("spline_It1", spline_It1 )
    coor_x = np.arange(x)
    coor_y = np.arange(y)

    spline_It = RectBivariateSpline(coor_x, coor_y, It)
#    print("spline_It", spline_It)
    p = p0
    while np.sum(np.multiply(p_del,p_del)) >= threshold: 
        ref1 = x2 + 0.1
        xrange = np.arange(x1, ref1)
        ref2 = y2 + 0.1
        yrange = np.arange(y1, ref2)
        mesh_x, mesh_y = np.meshgrid(xrange, yrange)

        warped = spline_It.ev(mesh_y, mesh_x)
        f2 = warped.flatten()
        
        sum_x1 = x1 +p[0]
        sum_x2 = x2 +0.1+p[0]
        range_x= np.arange(sum_x1, sum_x2)
        sum_y1 = y1+p[1]
        sum_y2 = y2+0.1+p[1]
        range_y = np.arange(sum_y1, sum_y2)
        grid_x, grid_y = np.meshgrid(range_x, range_y)
        orig = spline_It1.ev(grid_y, grid_x)
        f1 = orig.flatten()
        
        grad_dy = spline_It1.ev(grid_y, grid_x, dx=1, dy=0)
        grad_dy = grad_dy.flatten()

        grad_dx = spline_It1.ev(grid_y, grid_x, dx=0, dy=1)
        grad_dx = grad_dx.flatten()

        matrix_1 = np.zeros((grad_dx.shape[0], 2))
        matrix_1[:, 0], matrix_1[:, 1] = grad_dx, grad_dy
        matrix_2 = f2 - f1
        print("b", matrix_2)
        dot_prod1 = np.dot(np.transpose(matrix_1), matrix_2)
        dot_prod2 = np.dot(np.transpose(matrix_1), matrix_1)
        inverse = np.linalg.inv(dot_prod2)

        p_del = np.dot(inverse, dot_prod1)
        p_del = p_del.flatten()
        p = p + p_del

    return p
