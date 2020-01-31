import numpy as np
import threading
from multiprocessing import Pool
import queue
import imageio
import os,time
import math
import visual_words

def recog_for_one(params):
    index, image_path, label= params
    #params = para.split(",")
    dictionary = np.load("dictionary.npy")
    dictionary_size = dictionary.shape[0]
    SPM_layer_num = 3
    feats = get_image_feature(os.path.join("../data/",image_path), dictionary, SPM_layer_num, dictionary_size)
    feature = np.reshape(feats,(1,-1))
    np.savez("../refer"+"feat_one"+str(index)+".npz", feature=feature, label=label)
    return
    
def build_recognition_system(num_workers=4):
    dictionary = np.load("dictionary.npy")
    train_set = np.load("../data/train_data.npz")
    dictionary_size = dictionary.shape[0]
    SPM_layer_num = 3
    N = train_set['files'].shape[0]
    a = ((4**(SPM_layer_num-1)) - 1)/3
    train_feat = np.zeros((1, int(dictionary_size*a)))
    train_labels = []
    if not os.path.exists("../refer"):
        os.mkdir("../refer")
    params = zip(list(range(0,N)), train_set['files'], train_set['labels'])
    pool = Pool(num_workers)
    pool.map(recog_for_one, params)
    for i in range(1,N):
        reference =np.load("../refer"+"feat_one"+str(i)+".npz")
        train_features= np.append(train_feat, reference['feature'])
        train_labels = np.append(train_labels, reference['label'])
    np.savez("trained_system.npz", train_features= train_features, train_labels= train_labels, dictionary=dictionary, SPM_layer_num=SPM_layer_num)

def eval_for_one(image_path, dictionary, SPM_layer_num, train_features, index):
    test_set = np.load("../data/test_data.npz")
    #image_path = os.path.join("../data/",test_set[test_set.files[0]][arg])
    img = imageio.imread(image_path)
    SPM_layer_num = 3
    trained_system = np.load("trained_system.npz")
    train_features = trained_system['train_features']
    dictionary = np.load("dictionary.npy")
    
    if len(img.shape) == 2:     
        img = np.tile(img[:,np.expand_dims],(1,1,3))
    dictionary_size = dictionary.shape[0]
    wordmap = visual_words.get_visual_words(img, dictionary)
    hist = get_feature_from_wordmap_SPM(wordmap, SPM_layer_num, dictionary_size)
    sim = distance_to_set(hist, train_features)
    guessed_label = np.argmax(sim)
    np.savez("../refer/"+"guessed_label"+str(index)+".npz", guessed_label = guessed_label, index = index, image_path = image_path)
    return
    
def evaluate_recognition_system(num_workers=2):
    test_set = np.load("../data/test_data.npz")
    test_count = test_set['files'].shape[0]
    trained_system = np.load("trained_system.npz")
    train_features = trained_system['train_features']
    train_labels = trained_system['train_labels']
    dictionary = trained_system['dictionary']
    dictionary_size = dictionary.shape[0]
    SPM_layer_num = int(trained_system['SPM_layer_num'])
    if not os.path.exists("../refer"):
        os.mkdir("../refer")
    args = []   
    for i in range(test_count):
        img_path = os.path.join("../data/",test_set[test_set.files[0]][i])
        arg = (img_path, dictionary, SPM_layer_num, train_features, i)
        args.append(arg)
    #arg = list(range(test_count))
    pool = Pool(num_workers)
    pool.starmap(eval_for_one, args)
    conf_matrix = np.zeros((8,8))      
    for i in range(0, test_count):
        img_path = os.path.join("../refer/"+"guessed_label"+str(i)+".npy")
        guessed_label = img_path['guessed_label']
        a = test_set['labels'][1][i]
        b =  int(guessed_label)
        conf_matrix[a,b] = conf_matrix[a, b] + 1
    acc = np.sum(np.diagonal(conf_matrix))/np.sum(conf_matrix)
    print(acc)

def get_image_feature(file_path,dictionary,layer_num, dictionary_size):
    image = imageio.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    hist= get_feature_from_wordmap_SPM(wordmap, layer_num, dictionary_size)
    return hist
          
def distance_to_set(word_hist,histograms):
    #wordmap = visual_words.get_visual_words(image, dictionary)
    #hist= get_feature_from_wordmap_SPM(wordmap, layer_num, dictionary_size)
    minima = np.minimum(word_hist, np.transpose(histograms))
    sim = np.sum(minima, axis=1)
    return sim

def get_features_from_wordmap(wordmap, dictionary_size):
    histogram,cells = np.histogram(wordmap, bins=dictionary_size, density=True)
    return histogram, cells

def get_feature_from_wordmap_SPM(wordmap,layer_num,dictionary_size):
    h = wordmap.shape[0]
    w = wordmap.shape[1]
    hist_all =[]
    S = layer_num-1
    for m in range(S,-1,1):
        for i in range(0,2**(m-1)):
            for j in range(0,2**(m-1)):
                if m==0 or m==1:
                    weight = 2**(-S)
                else:
                    weight = 2**(m-S-1)
                first_index = math.floor(((h/2**m)*i)+1)
                second_index = math.floor((h/2**m)*(i+1))
                third_index = math.floor((w/2**m)*j+1)
                fourth_index = math.floor((w/2**m)*(j+1))
                hist_all =[h,(weight/(2**(2*m))) * get_image_feature(wordmap(range(first_index,second_index),range(third_index,fourth_index)),dictionary_size)]
    return hist_all