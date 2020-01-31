import numpy as np
import scipy.ndimage.morphology
from LucasKanadeAffine import LucasKanadeAffine
from scipy.interpolate import RectBivariateSpline

from InverseCompositionAffine import InverseCompositionAffine


def SubtractDominantMotion(image1, image2):
    N = image1.shape
    mask = np.zeros(N, dtype=bool)
    threshold = 0.1
    x1 = image1.shape[0]
    y1 = image1.shape[1]
    x2 = image2.shape[0]
    y2 = image2.shape[1]
    
    struct = np.array(([0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]))
    coor_x1 = np.arange(x1)
    # Classic approach
    spline1 = RectBivariateSpline(coor_x1, np.arange(y1), image1)
    spline2 = RectBivariateSpline(np.arange(x2), np.arange(y2), image2)
#    M = LucasKanadeAffine(image1, image2)
    
    # Inverse Composition Methop
    M = InverseCompositionAffine(image1, image2)

    range_x = np.arange(0, image2.shape[1])
    range_y = np.arange(0, image2.shape[0])
    grid_x, grid_y = np.meshgrid(range_x, range_y)

    n_1 = np.dot(M[1, 0],grid_x)
    n_2 = np.dot(M[1, 1],grid_y)
    const2 = M[1, 2]
    posit_y =n_1 + n_2 + const2
    warped2 = spline2.ev(grid_y, grid_x)
    
    print("posit_y", posit_y)
    m_1 = np.dot(M[0, 0],grid_x)
    m_2 = np.dot(M[0, 1], grid_y)
    const1 = M[0, 2]
    posit_x = m_1 + m_2 + const1
    warped1 = spline1.ev(posit_y, posit_x)
        
    print("posit_x", posit_x)
    desired_region = np.zeros(N)
    desired_region[10:150, 60:220] = 1.0

    not_required = []
    if posit_x.any() < 0:
        not_required = True
    elif posit_x.any() >= y1:
        not_required = True
    elif posit_y.any() < 0:
        not_required = True
    elif posit_y.any() >= x1:
        not_required = True
    warped1[not_required], warped2[not_required] = 0, 0

    subtract = abs(warped2 - warped1)*desired_region
    mask = (subtract > threshold) & (warped2 != 0)
    mask = scipy.ndimage.morphology.binary_dilation(mask, struct)
    return mask