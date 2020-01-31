import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

my_device = torch.device('cpu')

# initialise the variables
my_batch_size = 32
my_learning_rate = 0.01
my_hidden_size = 64
max_iters = 50

# initialise the data
my_training_data = scipy.io.loadmat('./data/data1/nist36_train_set1.mat')
my_testing_data = scipy.io.loadmat('./data/data1/nist36_test_set1.mat')
train_x, train_y = my_training_data['train_data'], my_training_data['train_labels']
test_x, test_y = my_testing_data['test_data'], my_testing_data['test_labels']

# load the data and set type
train_x_ts = torch.from_numpy(train_x).type(torch.float32)
train_y_ts = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=my_batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=my_batch_size, shuffle=False)

# define the class
class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# initailise the lists
train_loss = []
test_accuracy = []

# initialise the object
my_model = NeuralNetwork(train_x.shape[1], my_hidden_size, train_y.shape[1])

optimizer = torch.optim.SGD(my_model.parameters(), lr=my_learning_rate, momentum=0.9)
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
        correct += predicted.eq(targets.data).cpu().sum().item()

        # execute backward()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/train_y.shape[0]
    train_loss.append(total_loss)
    
    # append elements to test_accuracy
    test_accuracy.append(acc)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr, total_loss, acc))

plt.figure('accuracy')
plt.plot(range(max_iters), test_accuracy, color='g')
plt.legend(['train accuracy'])
plt.show()

plt.figure('loss')
plt.plot(range(max_iters), train_loss, color='g')
plt.legend(['train loss'])
plt.show()

print('Train accuracy: {}'.format(test_accuracy[-1]))

torch.save(my_model.state_dict(), "q7_1_1.pkl")


# To run on the test data
test_correct = 0

for data_ele in test_loader:
    # To get the inputs
    inputs = torch.autograd.Variable(data_ele[0])
    
    labels = torch.autograd.Variable(data_ele[1])
    targets = torch.max(labels, 1)[1]

    # To get the output
    y_pred = my_model(inputs)
    
    loss = nn.functional.cross_entropy(y_pred, targets)
    predicted = torch.max(y_pred, 1)[1]
    test_correct += predicted.eq(targets.data).cpu().sum().item()

test_accuracy = test_correct/test_y.shape[0]

print('Test accuracy: {}'.format(test_accuracy))