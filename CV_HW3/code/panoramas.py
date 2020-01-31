import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import matplotlib.pyplot as plt

def get_warped_min_max(im2, H2to1):

    (rows, cols, _) = np.shape(im2)

    x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, cols -1, 0, 0, rows -1, cols-1, rows -1

    im2_warped_homogeneous = np.matmul(H2to1, np.array([[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]]))
    im2_warped = im2_warped_homogeneous / im2_warped_homogeneous [2,:]
    im2_warped = im2_warped[0:2,:]

    min_x, max_x,  min_y, max_y = np.min(im2_warped[0,:]), np.max(im2_warped[0,:]), np.min(im2_warped[1,:]), np.max(im2_warped[1,:]) 
    
    return min_x, max_x, min_y, max_y

def imageStitching(im1, im2, H2to1):
    warp_im2 = cv2.warpPerspective(im2, H2to1, (im1.shape[1]*2, im1.shape[0]))
    
    im1_withzeros = np.zeros_like(warp_im2)
    im1_withzeros[:, :im1.shape[1], :] = im1
    pano_im = np.maximum(im1_withzeros, warp_im2)
    
    return pano_im

def imageStitching_noClip(im1, im2, H2to1):

    # I kept width as 1200.. It can be changed.. 
    givenWidth=1200

    min_x_1, max_x_1, min_y_1, max_y_1 = get_warped_min_max(im1, H2to1=np.eye(3))
    min_x_2, max_x_2, min_y_2, max_y_2 = get_warped_min_max(im2, H2to1)

    max_x, min_x, max_y, min_y = max(max_x_1, max_x_2), min(min_x_1, min_x_2), max(max_y_1, max_y_2),  min(min_y_1, min_y_2)
    
    width = max_x - min_x
    scale = givenWidth / width

    M = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    warp_matrix = np.matmul(M, H2to1)

    min_x_1, max_x_1, min_y_1, max_y_1 = get_warped_min_max(im1, M)
    min_x_2, max_x_2, min_y_2, max_y_2 = get_warped_min_max(im2, warp_matrix)

    max_x, min_x, max_y, min_y = max(max_x_1, max_x_2), min(min_x_1, min_x_2), max(max_y_1, max_y_2), min(min_y_1, min_y_2)

    ty, tx = -min_y, -min_x
    M = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])

    warp_matrix = np.matmul(M, H2to1)

    min_x_1, max_x_1, min_y_1, max_y_1 = get_warped_min_max(im1, M)
    min_x_2, max_x_2, min_y_2, max_y_2 = get_warped_min_max(im2, warp_matrix)
    max_y, min_y = max(max_y_1, max_y_2), min(min_y_1, min_y_2)
    
    out_size = (1200, int(max_y - min_y))
    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, warp_matrix, out_size)

    pano_im = np.maximum(warp_im1, warp_im2)
    return pano_im



def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
    
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im
    
if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)
    
    pano_im1 = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/q6_1.jpg', pano_im1)
    
    pano_im2 = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_2.jpg', pano_im2)
    

    pano_im3 = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', pano_im3)
    cv2.imshow('panoramas', pano_im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

