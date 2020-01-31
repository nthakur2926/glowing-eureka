import numpy as np
import numpy.linalg as p
import numpy.matlib
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):

    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)

    N = np.shape(p1)[1]
   
    onesColumn = np.ones((N,1))
    zeros3ByN = np.zeros((N,3))
    x = p1.T[:,0][:,np.newaxis]
    y = p1.T[:,1][:,np.newaxis]
    A1 = np.hstack((p2.T, onesColumn, zeros3ByN, -(x * p2.T), -x))
    A2 = np.hstack((zeros3ByN, p2.T, onesColumn, -(y * p2.T), -y))

    A = np.vstack((A1, A2))

    # TODO check full_matrices
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    
    # Last row of vh corresponds to eigen vector with least eigen value
    H2to1 = vh[-1,:].reshape((3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    H2to1 = np.zeros((3, 3))
    max_inliers = -1
    bestH = np.zeros((3,3))
    number_of_matches = matches.shape[0]
    print("Number of matches: ", number_of_matches)

    p1= np.zeros((number_of_matches, 2))
    p2 = np.zeros((number_of_matches, 2))
    p1[:, :] = locs1[matches[:, 0], 0:2]
    np.transpose(p1)
    p2[:, :] = locs2[matches[:, 1], 0:2]
    np.transpose(p2)
    inlier_indices = None
    for iter in range(num_iter):
        rand_index = np.random.choice(number_of_matches, 4)
        index1 = p1[rand_index,: ]
        index2 = p2[rand_index, :]

        H = computeH(index1.T, index2.T)
        # Compute number of inliers
        X = np.append(p2.T, np.ones((1, number_of_matches)), axis=0)
#        print(X.shape)
        estimated_p1 = np.matmul(H, X)
#        f = np.matlib.repmat(estimated_p1[2,:],2,1)
        estimated_p1 = estimated_p1/ estimated_p1[2,:]
 
        error_x = (p1.T[0,:] - estimated_p1[0,:])**2
        error_y = (p1.T[1,:] - estimated_p1[1,:])**2
        error = error_x + error_y
#        print("error", error)
        inliers = np.where(error < tol**2)[0]
        num_inliers = len(inliers)
#        print("num_inliers, max_inliers")
#        print(num_inliers, max_inliers)
        if num_inliers > max_inliers:
            bestH = H
            max_inliers = num_inliers
            inlier_indices = inliers
#            print(num_inliers, max_inliers)
#    print("inliers",len(inliers))
    H2to1 = computeH(p1[inlier_indices,:].T, p2[inlier_indices,:].T)
    np.save('../results/bestH.npy', H2to1)
    print("RANSAC max number of inliers: ", max_inliers)
    return H2to1
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
#    im2 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    print(desc1.shape, desc2.shape)
    matches = briefMatch(desc1, desc2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

