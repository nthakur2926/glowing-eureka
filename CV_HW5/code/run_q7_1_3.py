import numpy as np
import torch
import torch.nn as nn
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt

device = torch.device('cpu')

# initialise the variables
max_iters = 20
my_batch_size = 64
my_learning_rate = 0.01

# initialise the data
my_train_data = scipy.io.loadmat('./data/data1/nist36_train_set1.mat')
my_test_data = scipy.io.loadmat('./data/data1/nist36_test_set1.mat')

my_train_x, my_train_y = my_train_data['train_data'], my_train_data['train_labels']
test_x, test_y = my_test_data['test_data'], my_test_data['test_labels']

# define n-dimensional array and reshape
my_train_x = np.array([np.reshape(x, (32, 32)) for x in my_train_x])
test_x = np.array([np.reshape(x, (32, 32)) for x in test_x])

train_x_ts = torch.from_numpy(my_train_x).type(torch.float32).unsqueeze(1)
train_y_ts = torch.from_numpy(my_train_y).type(torch.LongTensor)

# load the data
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=my_batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32).unsqueeze(1)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)

test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=my_batch_size, shuffle=False)

# define the class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(16*16*16, 1024), nn.ReLU(),nn.Linear(1024, 36))
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(), nn.MaxPool2d(stride=2, kernel_size=2))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 16*16*16)
        x = self.fc1(x)
        return x

# initailise the lists
train_loss = []
test_accuracy = []

# initialise the object
my_model = Net()

optimizer = torch.optim.SGD(my_model.parameters(), lr=my_learning_rate)

for itr in range(max_iters):
    total_loss = 0
    correct = 0
    
    for data_ele in train_loader:
        # To get the inputs
        inputs = torch.autograd.Variable(data_ele[0])
        
        labels = torch.autograd.Variable(data_ele[1])
        targets = torch.max(labels, 1)[1]

        # To get the output
        y_pred = my_model(inputs)
        
        loss = nn.functional.cross_entropy(y_pred, targets)
        
        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == targets.data).item()

        # execute backward()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/my_train_x.shape[0]
    
    # append data to train_loss
    train_loss.append(total_loss)
    
    # append data to test_accuracy
    test_accuracy.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

# plot the graph for train accuracy
plt.figure('accuracy')

plt.plot(range(max_iters), test_accuracy, color='g')
plt.legend(['train accuracy'])

plt.show()

# plot the graph for train loss
plt.figure('loss')

plt.plot(range(max_iters), train_loss, color='g')
plt.legend(['train loss'])

plt.show()

# show the result for Train accuracy
print('Train accuracy: {}'.format(test_accuracy[-1]))

torch.save(my_model.state_dict(), "q7_1_3.pkl")

# To run on the test data
test_correct = 0
for data in test_loader:
    # To get the inputs
    inputs = torch.autograd.Variable(data[0])
    
    labels = torch.autograd.Variable(data[1])
    targets = torch.max(labels, 1)[1]

    # To get the output
    y_pred = my_model(inputs)
    
    loss = nn.functional.cross_entropy(y_pred, targets)

    predicted = torch.max(y_pred, 1)[1]
    test_correct += torch.sum(predicted == targets.data).item()

# calculate test accuracy
test_accuracy = test_correct/test_x.shape[0]

# show the result for Test accuracy
print('Test accuracy: {}'.format(test_accuracy))