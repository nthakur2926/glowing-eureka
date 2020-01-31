import sys
import numpy as np
import scipy.ndimage as sni

def alignChannels(red, green, blue):
    
    min_index_green = sys.maxsize
    min_index_red = sys.maxsize
    
    green_row = []
    green_col = []
    red_row = []
    red_col = []
    
    for i in range(-30,30):
        for j in range(-30,30):
            green_updated = sni.shift(green, [i,j])
            ssd_GreenBlue = np.sum((blue - green_updated)**2)
            if ssd_GreenBlue < min_index_green:
                min_index_green = ssd_GreenBlue
                [green_row, green_col] = [i,j]
                
                
            red_updated = sni.shift(red, [i,j])
            ssd_RedBlue = np.sum((blue - red_updated)**2)
            if ssd_RedBlue < min_index_red:
                min_index_red = ssd_RedBlue
                [red_row, red_col] = [i,j]
                
    rgbResult = np.zeros((blue.shape[0], blue.shape[1], 3, ))
    rgbResult[...,0] = blue
    rgbResult[...,1] = sni.shift(green, [green_row, green_col])
    rgbResult[...,2] = sni.shift(red, [red_row, red_col])
    
    return rgbResult
    
            
    