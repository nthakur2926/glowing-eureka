import cv2
import numpy as np 
#import matplotlib.pyplot as plt 
#import pdb
#import argparse


def myHough(img_name,ce_params,hl_params):
    
    for line in hl_params:
        x_1, y_1, x_2, y_2 = line[0]
        cv2.line(img_name, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)

if __name__=="__main__":

    img_name = cv2.imread('img05.jpg')
    gray = cv2.cvtColor(img_name,cv2.COLOR_BGR2GRAY)
    ce_params = cv2.Canny(gray,150,200,apertureSize = 3)
    hl_params = cv2.HoughLinesP(ce_params,1,np.pi/180,40, minLineLength = 40, maxLineGap = 4)
    myHough(img_name, ce_params,hl_params)
    cv2.imshow('image', img_name)
    cv2.imwrite('img05_hlines.jpg', img_name)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()