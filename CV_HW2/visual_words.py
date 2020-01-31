import numpy as np
from multiprocessing import Pool
import imageio
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as l
import math
from skimage import color
import sklearn.cluster
import torch 
import os,time
import util
import random
import cv2

def extract_filter_responses(image):
    M, N = image.shape[0], image.shape[1]
    scales = np.array([1, 2, 4, 8, 8 *(math.sqrt(2))])
    fr = np.zeros((M,N,0))
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 

    if len(image.shape) == 2:     
        image = np.tile(image[:,np.expand_dims],(1,1,3))
    image = color.rgb2lab(image) 
    
    for i in range(scales.shape[0]):
        fr = np.concatenate((fr, l.filters.gaussian_filter(image, sigma=(scales[i], scales[i], 0))),axis = 2)
        red = l.filters.gaussian_laplace(image[:,:,0], scales[i])
        green = l.filters.gaussian_laplace(image[:,:,1], scales[i])
        blue = l.filters.gaussian_laplace(image[:,:,2], scales[i])
        fr = np.concatenate((fr, np.stack((red,green,blue), axis = -1)), axis = 2)
        fr = np.concatenate((fr, l.filters.gaussian_filter(image, sigma=(0, scales[i], 0),order =1)), axis = 2)
        fr = np.concatenate((fr, l.filters.gaussian_filter(image, sigma=(scales[i], 0, 0),order =1)), axis = 2)

    return fr
        
def get_visual_words(image,dictionary):
    
    collected_fr = extract_filter_responses(image)
    m,n = collected_fr.shape[0],collected_fr.shape[1]
    f = collected_fr.shape[-1]
    collected_fr = collected_fr.reshape(m*n, f)
    distances = scipy.spatial.distance.cdist(collected_fr, dictionary)
    wordmap = np.argmin(distances, axis=1)
    wordmap = wordmap.reshape(m,n)
    print(wordmap)
    return wordmap

def compute_dictionary_one_image(args):
      
    i,alpha,image_path = args
    image = imageio.imread(os.path.join("../data/",image_path))
    fr = extract_filter_responses(image)
    print(fr)
    x = np.random.choice(fr.shape[0], alpha, 1)#torch.multinomial(fr.shape[0], alpha, replacement=True, out=None)
    print(x)
    y = np.random.choice(fr.shape[1], alpha, 1)#torch.multinomial(fr.shape[1], alpha, replacement=True, out=None)
    print(y)
    sampled_response = fr[x,y,:]
    if not os.path.exists("../results"):
        os.mkdir("../results")
    np.savez("../results/"+str(i)+".npz", res = sampled_response)

def compute_dictionary(num_workers=4):
    K = 200
    alpha = 200   
    training_data = np.load("../data/train_data.npz")
    
    x = training_data['files'].shape[0]
    #x = 2
    args = zip(list(range(x)), [alpha]*x, training_data['files'])
    
    print(training_data['files'])

    pool = Pool(num_workers)
    pool.map(compute_dictionary_one_image, args)
    
    print("done")
    for i in range(x):
        collected_fr = np.load("../results/"+str(i)+".npz")
        print(collected_fr["res"].shape)
        if i == 0:
            fr = collected_fr["res"]
        else:
            fr = np.vstack([fr, collected_fr["res"]])
        #fr.append(collected_fr['fr'])
    kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(fr)
    dictionary = kmeans.cluster_centers_
    np.save("dictionary.npy", dictionary)

  