import time
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dimensions, hidden_size)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        # self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        # self.relu2 = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = inputs
        t = self.fc1(x)
        t = self.relu1(t)
        # t = self.fc2(t)
        # t = self.relu2(t)
        y = self.shape_outputs(t)
        return y


def preprocess(raw_data):
    X = raw_data.drop('RMR', axis=1)
    y = raw_data.RMR
    X_train, X_test, y_train, y_test = train_test_split(df_to_tensor(X), df_to_tensor(y), test_size=0.2, random_state=42)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    y_train = normalize_data(y_train, target=True)
    y_test = normalize_data(y_test, target=True)

    return X_train, X_test, y_train, y_test


def df_to_tensor(df):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return torch.from_numpy(df.values).float().to(device)


def normalize_data(data, target=False):
    data = data.numpy()
    if not target:
        for i in range(data.shape[1]):
            mean = np.mean(data[:, i])
            sigma = np.std(data[:, i])
            data[:, i] = data[:, i] - mean
            data[:, i] /= sigma
        return data
    elif target:
        mean = np.mean(data)
        std = np.std(data)
        data -= mean
        data /= std
        return data


def train(X_train, X_test, y_train, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # Hyper Parameters
    batch_size = len(X_train)
    num_epochs = 2500
    learning_rate = 0.0001
    size_hidden = 1024

    batch_no = len(X_train) // batch_size  # number of batches per epoch

    # Save data for plotting
    epoch_list = []
    loss_list = []
    test_loss_list = []

    # Set model
    model = Model(X_train[0].shape[0], size_hidden)
    model.to(device)
    # print(f"Model architecture is: {model}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("total trainable parameters: {}".format(pytorch_total_params))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # Train loop
    running_loss = 0.0
    for epoch in range(num_epochs):
        # set model in train mode
        model.train()
        # shuffle data
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            inputs = Variable(torch.FloatTensor(X_train[start:end]))
            labels = Variable(torch.FloatTensor(y_train[start:end]))
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(inputs)
            # calculate loss
            loss = loss_function(outputs, torch.unsqueeze(labels, dim=1))
            # the three steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute test loss
        model.eval()
        test_inputs = Variable(torch.FloatTensor(X_test))
        test_labels = Variable(torch.FloatTensor(y_test))
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)
        test_outputs = model(test_inputs)
        test_loss = loss_function(test_outputs, torch.unsqueeze(test_labels, dim=1))
        test_loss = test_loss.item()

        # Print progress
        print('Epoch {}'.format(epoch + 1), "loss: ", np.round(running_loss, 6), "test loss: ", np.round(test_loss, 6))
        # Save data for plotting
        loss_list.append(running_loss)
        test_loss_list.append(test_loss)
        epoch_list.append(epoch)
        running_loss = 0.0

    plt.plot(epoch_list, loss_list, label="train loss")
    plt.plot(epoch_list, test_loss_list, label="test loss")
    plt.xlabel("No. of epoch")
    plt.ylabel("Loss")
    plt.title(f"Epochs vs Loss\nModel: Hidden size: {size_hidden} | opt: {'SGD'} | lr: {learning_rate} | batch size: {batch_size} | Loss function: {'MSE'}")
    plt.legend()
    plt.show()
