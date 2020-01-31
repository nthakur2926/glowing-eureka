import numpy as np
#import numpy.linalg as f
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    h = gaussian_pyramid.shape[0]
    w = gaussian_pyramid.shape[1]
    DoG_pyramid = np.zeros((h, w, 0))  
    DoG_levels = levels[1:]
    for i in range(1, len(DoG_levels)+1):
        diff = gaussian_pyramid[:,:,i] - gaussian_pyramid[:,:,i-1]
        DoG_pyramid = np.concatenate((DoG_pyramid, np.expand_dims(diff, axis=2)),axis =2)
        #DoG_pyramid = np.concatenate((DoG_pyramid,diff[:,:,np.newaxis]),axis =2)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    principal_curvature = None
    x = DoG_pyramid.shape[0]
    y = DoG_pyramid.shape[1]
    z = DoG_pyramid.shape[2]
    principal_curvature = np.zeros((x, y, z))
    for i in range(z):
        
        level = DoG_pyramid[:,:,i]
        image = DoG_pyramid[:,:,i]
#        [Dx, Dy] = np.gradient(level)
#        [Dxx, Dxy] = np.gradient(Dx)
#        [Dyx, Dyy] = np.gradient(Dy)
        Dx = cv2.Sobel(image,ddepth = -1,dx=1, dy=0)
        Dy = cv2.Sobel(image,ddepth = -1,dx=0, dy=1)

        Dxx = cv2.Sobel(Dx,ddepth = -1,dx=1, dy=0)
        Dxy = cv2.Sobel(Dx,ddepth = -1,dx=0, dy=1)
        Dyx = cv2.Sobel(Dy,ddepth = -1,dx=1, dy=0)
        Dyy = cv2.Sobel(Dy,ddepth = -1,dx=0, dy=1)
        trace_H = Dxx+Dyy
        determinant_H = (Dxx*Dyy)-(Dxy*Dyx)
        principal_curvature[:,:,i] = (trace_H *trace_H)/determinant_H
        #print(principal_curvature)
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    locsDoG = np.zeros((0, 3))
    m = DoG_pyramid.shape[0]
    n = DoG_pyramid.shape[1]
    p = DoG_pyramid.shape[2]
    for l in range(1,m-1):
        for x in range(1, n-1):
            for y in range(p):
                if x < 3 or x > n-2 or l < 3 or l > m-2:
                    continue
                if abs(DoG_pyramid[l,x,y]) < 0.03:
                    continue
                if abs(principal_curvature[l,x,y]) > 12:
                    continue
                arr1 = [-1, 0, 1]
                arr2 = [-1, 1]
                if (y == 0):
                    arr2 = [1]
                if y == p - 1:
                    arr2 = [-1]

                contours = []
                for dev1 in arr1:
                    for dev2 in arr1:
                        contours.append(DoG_pyramid[l+dev1, x+dev2, y])
                for dev3 in arr2:
                    contours.append(DoG_pyramid[l, x, y+dev3])

                contours = np.array(contours)
                minima = np.argmin(contours)
                maxima = np.argmax(contours)
                
                if minima==4 or maxima==4:
                    arg = np.array([x,l,y])
                    arg = arg.reshape(1,3)
                    locsDoG = np.append(locsDoG, arg, axis=0).astype(int)
                    #print("shape=",locsDoG.shape)

    return locsDoG

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    
    gaussian_pyramid = createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4])
    DoG_pyr, DoG_levels = createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4])
    principal_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, principal_curvature, th_contrast, th_r)
#    print("locshape",locsDoG.shape)
#    print("gauss",gaussian_pyramid.shape)
    return locsDoG, gaussian_pyramid

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
#    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
#    displayPyramid(DoG_pyr)
    # test compute principal curvature
    #pc_curvature = computePrincipalCurvature(DoG_pyr)
    # test get local extrema
#    th_contrast = 0.03
#    th_r = 12
#    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    print("locshape",locsDoG.shape)
    print("gauss",gaussian_pyramid.shape)
    x = im.shape[0]
    y = im.shape[1]
    b = locsDoG.shape[0]
    for i in range(b):
        point = locsDoG[i,:]
        coord_1 = point[0]
        coord_2 = point[1]      
        cv2.circle(im,(coord_1, coord_2), 1, color=(0,255,0), lineType=cv2.LINE_AA)
    cv2.namedWindow("rgb_img", cv2.WINDOW_NORMAL)
    im = cv2.resize(im,(y*5, x*5))
    cv2.imwrite('../results/keypoints.png', im)
    cv2.imshow('rgb_img', im)
    cv2.waitKey(0)


