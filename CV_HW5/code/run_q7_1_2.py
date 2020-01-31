import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device('cpu')

# initialise the variables
my_batch_size = 64
my_learning_rate = 0.01
max_iters = 3

# initialise the data
transform = transforms.Compose([transforms.ToTensor(),])
my_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# load the data
train_loader = torch.utils.data.DataLoader(my_train_set, batch_size=my_batch_size, shuffle=True, num_workers=0)
my_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(my_test_set, batch_size=my_batch_size, shuffle=False, num_workers=0)

# define the class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(14*14*4, 10))
        self.conv1 = nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1), 
                                   nn.ReLU(), nn.MaxPool2d(stride=2, kernel_size=2))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*4)
        x = self.fc1(x)
        return x

# initailise the lists
train_loss = []
test_accuracy = []

# initialise the object
model = Net()

optimizer = torch.optim.SGD(model.parameters(), lr=my_learning_rate)


for itr in range(max_iters):
    total_loss = 0
    correct = 0
    
    for data_ele in train_loader:
        # To get the inputs
        inputs = torch.autograd.Variable(data_ele[0])
        
        labels = torch.autograd.Variable(data_ele[1])

        # To get the output
        y_pred = model(inputs)
        
        loss = nn.functional.cross_entropy(y_pred, labels)

        total_loss += loss.item()
        predicted = torch.max(y_pred, 1)[1]
        correct += torch.sum(predicted == labels.data).item()

        # execute backward()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    acc = correct/len(my_train_set)
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

print('Train accuracy: {}'.format(test_accuracy[-1]))

torch.save(model.state_dict(), "q7_1_2.pkl")

# To run on the test data
test_correct = 0
for data_ele in test_loader:
    # To get the inputs
    inputs = torch.autograd.Variable(data_ele[0])
    
    labels = torch.autograd.Variable(data_ele[1])

    # To get the inputs
    y_pred = model(inputs)
    
    loss = nn.functional.cross_entropy(y_pred, labels)
    predicted = torch.max(y_pred, 1)[1]
    test_correct += torch.sum(predicted == labels.data).item()

# calculate test accuracy
test_accuracy = test_correct/len(my_test_set)

print('Test accuracy: {}'.format(test_accuracy))