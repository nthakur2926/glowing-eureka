import numpy as np
import scipy.io
from nn import *
import string
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('./data/data1/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('./data/data1/nist36_valid_set1.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.001
hidden_size = 64

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

# with default settings, you should get loss < 150 and accuracy > 80%
trainingLoss = []
trainingAccuracy = []
validation_loss = []
validation_accuracy = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss = total_loss + loss
        total_acc = total_acc + acc

        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    
    trainingAccuracy.append(total_acc)
    trainingLoss.append(total_loss)
    total_acc = total_acc/batch_num

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, total_acc))

    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1, params, 'output', softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    validation_loss.append(loss)
    validation_accuracy.append(acc)

plt.figure('Loss Plot')
plt.plot(range(max_iters), trainingLoss, color='b')
plt.plot(range(max_iters), validation_loss, color='r')
plt.legend(['Training', 'Validation'])
plt.show()

plt.figure('Accuracy Plot')
plt.plot(range(max_iters), trainingAccuracy, color='b')
plt.plot(range(max_iters), validation_accuracy, color='r')
plt.legend(['Training', 'Validation'])
plt.show()

# test set
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, probs)
print('Accuracy for Test set: ', test_acc)

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
print('Accuracy for Validation Set: ', valid_acc)

test_data = scipy.io.loadmat('./data/data1/nist36_test_set1.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']


if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8))

for i in range(0,64):
    res = np.reshape(params['Wlayer1'][:, i], (32, 32))
    grid[i].imshow(res)
    plt.axis('off')
plt.show()


# Q3.1.3

def con_mat(probs, y):
    confusion_matrix = np.zeros((y.shape[1], y.shape[1]))
    max_reshaped = np.expand_dims(np.max(probs, axis=1), axis=1)
    predicted_label = (probs == max_reshaped)
    summation = np.sum(predicted_label, axis=1)
    sanc = np.where(summation> 1)[0]
    N = sanc.shape[0]
    for i in range(0,N):
        s1 = np.max(predicted_label[i, :])
        labels = predicted_label[i, :]
        predicted_label[i, np.where(labels == s1)[0][0] + 1:] = False
        
    gap = y.shape[0]
    est = [np.where(y[i, :] == 1)[0][0] for i in range(0,gap)]
    est_var = [np.where(predicted_label[i, :] == 1)[0][0] for i in range(0,gap)]

    for alpha, beta in zip(est, est_var):
        confusion_matrix[alpha][beta] +=1
    return confusion_matrix

train_data = scipy.io.loadmat('./data/data1/nist36_train_set1.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
h1 = forward(train_x, params, 'layer1')
train_probs = forward(h1, params, 'output', softmax)

confusion_matrix = con_mat(train_probs, train_y)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()


valid_data = scipy.io.loadmat('./data/data1/nist36_valid_set1.mat')
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
h1 = forward(valid_x, params, 'layer1')
valid_probs = forward(h1, params, 'output', softmax)

confusion_matrix = con_mat(valid_probs, valid_y)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()

test_data = scipy.io.loadmat('./data/data1/nist36_test_set1.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
test_probs = forward(h1, params, 'output', softmax)

confusion_matrix = con_mat(test_probs, test_y)
plt.imshow(confusion_matrix, interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36), string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()



