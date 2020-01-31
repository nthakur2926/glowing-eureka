import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import matplotlib.pyplot as plt

def makeTestPattern(patch_width=9, nbits=256):
    compareX = np.random.randint(0,80,nbits)
    compareY = np.random.randint(0,80,nbits)
#    print("x",compareX)
#    print("x",len(compareX))
#    print("y",compareY)
#    print("y",len(compareY))
    return  compareX, compareY

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'

if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])

def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    h = im.shape[0]
    w = im.shape[1]
    m = locsDoG.shape[0]
    #print(locsDoG.shape)
    #print(locsDoG[0,:])
    locs = []
#    print("Compsre")
#    print(compareX)
    for k in range(m):
        keypoint_x=locsDoG[k,1]
        keypoint_y=locsDoG[k,0]
        levels = locsDoG[k,2]
#        print("x",keypoint_x)
#        print("y",keypoint_y)
        if (keypoint_x-4 >=0) and (keypoint_y-4 >= 0) and (keypoint_x + 4 < h) and (keypoint_y + 4 < w):
#            locs = np.append(locs,locsDoG[k,:])
            if locs == []:
                locs = locsDoG[k,:]
            else:
                locs = np.vstack((locs,locsDoG[k,:]))
            
#            print("bell",locs.shape)
    locs = np.array(locs)
#    print("a",locs.shape)
    #desc = np.zeros((0,256))
    desc = []
    for v in range(m):
#        print("index",m)
        keypoint_x=locsDoG[v,1]
        keypoint_y=locsDoG[v,0]
        levels = locsDoG[v,2]
        if (keypoint_x-4 >=0) and (keypoint_y-4 >= 0) and (keypoint_x + 4 < h) and (keypoint_y + 4 < w): 
            patch_arr = np.zeros((9,9))
            #patch_arr[i, :] = (im[keypoint_x-4+i,keypoint_y-4:keypoint_y+5] for i in range(9))
#            print("inside", m)
            for i in range(9):
                patch_arr[i, :] = im[keypoint_x-4+i,keypoint_y-4:keypoint_y+5]
            patch_arr.resize(81)
            descr = np.zeros((1,256))
            for k in range(0,256):
                #temp = patch_arr.flatten()
#                print(temp)
#                print(compareX[k])
                if patch_arr[compareX[k]] < patch_arr[compareY[k]]:
                    descr[0, k] = 1  
                else:
                    descr[0, k] = 0
            if desc == []:
                desc = descr
            else:
                desc = np.vstack((desc,descr))
#            desc = np.append(desc,descr,axis=0)
#            print("desc", desc.shape)
    return locs, desc

def briefLite(im):
    locsDoG, gaussian_pyramid = DoGdetector(im)
    k = np.sqrt(2)
    sigma0 = 1
    levels = [-1,0,1,2,3,4]
    th_contrast = 0.03
    th_r = 12
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255

    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY)
    print("uoi",locs.shape)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
    
    

if __name__ == '__main__':
    # test makeTestPattern
    #compareX, compareY = makeTestPattern()
    # load test pattern for Brief
#    test_pattern_file = '../results/testPattern.npy'
#    
#    if os.path.isfile(test_pattern_file):
#        # load from file if exists
#        compareX, compareY = np.load(test_pattern_file)
#    else:
#        # produce and save patterns if not exist
#        compareX, compareY = makeTestPattern()
#        if not os.path.isdir('../results'):
#            os.mkdir('../results')
#        np.save(test_pattern_file, [compareX, compareY])

    
    # test briefLite
    im = cv2.imread('../data/pf_scan_scaled.jpg')
    locs, desc = briefLite(im)
    #print(locs.shape)
    fig = plt.figure()
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    plt.plot(locs[:,0], locs[:,1], 'r.')
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    # test matches
    #im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    im2 = cv2.imread('../data/pf_stand.jpg')
#    im2 = cv2.imread('../data/model_chickenbroth.jpg')
    #im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)
