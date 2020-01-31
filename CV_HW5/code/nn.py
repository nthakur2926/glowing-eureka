import numpy as np
from util import *

# Q 2.1

def initialize_weights(in_size,out_size,params,name=''):
    total = out_size + in_size
    upper_val = np.sqrt(6.0 / total)
    lower_val = -upper_val
    
    W = np.random.uniform(lower_val, upper_val, (in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1

def sigmoid(x):
    res = 1.0 / (1.0 + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    
    pre_act, post_act = None, None
    W = params['W' + name]
    b = params['b' + name]

    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)
    params['cache_' + name] = (X, pre_act, post_act)
    return post_act

# Q 2.2.2 
def softmax(x):
    i = x.shape
    maximum_x = np.max(x, axis=1)
    res = np.zeros(i)
    shaped = np.expand_dims(maximum_x, axis=1)
    x_moved = x + (-1.0 * shaped)
    out = np.exp(x_moved)
    sum_out = np.sum(out, axis=1)
    res = out/np.expand_dims(sum_out, axis =1)
    return res
    
# Q 2.2.3
# compute total loss and accuracy
def compute_loss_and_acc(y, probs):
    y = y.astype(int)
    N = y.shape[0]
    y_predicted = probs == np.expand_dims(np.max(probs, axis=1),axis=1)
    sum_ypred = np.sum(y_predicted, axis=1)
    equal_prob = np.where(sum_ypred > 1)[0]
    logarithm = np.log(probs)
    loss = -1.0 * np.sum(y*logarithm)
    m = equal_prob.shape[0]
    for i in range(0,m):
        y_predicted[i, np.where(y_predicted[i, :]==np.max(y_predicted[i, :]))[0][0]+1:] = False
    dev = np.abs(y_predicted-y)
    acc = (N - (np.sum(dev))//2) / N
    
    return loss, acc 

def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res
# Q2.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    grad_X, grad_W, grad_b = None, None, None
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    
    grad_X = np.zeros(X.shape)
    grad_b = np.zeros(b.shape)
    grad_W  = np.zeros(W.shape)
    m = delta.shape[0]
    n = np.ones((1, m))
    dim_del = delta.ndim
    dim_x = X.ndim
    
    if dim_del == 1:
        delta = delta.reshape((1, m))
    if dim_x == 1:
        X = X.reshape((1, X.shape[0]))

    res = delta * activation_deriv(post_act)
    grad_X = np.dot(res, np.transpose(W))
    grad_W = np.dot(np.transpose(X), res)
    grad_b = np.dot(n, res).reshape([-1])
    
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X


# Q 2.4
def get_random_batches(x,y,batch_size):
    batches = []
    f = range(x.shape[0])
    m = len(f)
    
    while m > 0:
        choice = []
        updated_x = []
        updated_y = []
        batchID = np.random.randint(0,m,batch_size)
        for item in batchID:
            choice.append(f[item])
        for item in choice:
             updated_x.append(x[item])
        for item in choice:
            updated_y.append(y[item])
        x_array = np.array(updated_x)
        y_array = np.array(updated_y)
        batches.append((x_array, y_array))
        out = set(f) - set(choice)
        f = list(out)
        m = len(f)
    return batches
