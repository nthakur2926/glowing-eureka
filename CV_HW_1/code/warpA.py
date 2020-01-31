import numpy as np
from numpy.linalg import inv

def warp(im, A, output_shape):
    
    num_rows = output_shape[0]
    num_cols = output_shape[1]
    im_morphed = np.zeros_like(im)
    
    inverse_of_A = inv(A)
    
    for i in range(num_rows):
        for j in range(num_cols):
            p_warped = np.array([i, j ,1])
            p_destination = np.transpose(p_warped)
            p_source = np.dot(inverse_of_A,p_destination)
            s = np.round(p_source)
            
            x_morph = int(s[0])
            y_morph = int(s[1])
            
            if x_morph >= 0 and x_morph < num_rows and y_morph >= 0 and y_morph < num_cols:
                im_morphed[i][j] = im[x_morph][y_morph]
            else: 
                im_morphed[i][j] = 0                 

    return im_morphed
