import datetime
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics

target_mean = 0
target_std = 0


class Model(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_size, dropout):
        super().__init__()
        # Layer # 1
        self.fc1 = torch.nn.Linear(input_dimensions, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)
        # Layer # 2
        # self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # self.relu2 = torch.nn.ReLU()
        # self.dropout2 = torch.nn.Dropout(dropout)

        self.shape_outputs = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = inputs
        t = self.fc1(x)
        t = self.relu1(t)
        # t = self.fc2(t)
        # t = self.relu2(t)
        y = self.shape_outputs(t)
        return y


def df_to_tensor(df):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    return torch.from_numpy(df.values).float().to(device)


def normalize_data(data, target=False):
    global target_mean, target_std
    data = data.numpy()
    if not target:
        for i in range(data.shape[1]):
            if i > 7:
                continue
            mean = np.mean(data[:, i])
            sigma = np.std(data[:, i])
            data[:, i] = data[:, i] - mean
            data[:, i] /= sigma
        return data
    elif target:
        target_mean = np.mean(data)
        target_std = np.std(data)
        data -= target_mean
        data /= target_std
        return data


def preprocess(raw_data):
    X = raw_data.drop('RMR', axis=1)
    y = raw_data.RMR

    X_train, X_test, y_train, y_test = train_test_split(df_to_tensor(X), df_to_tensor(y), test_size=0.25, random_state=42)

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    y_train = normalize_data(y_train, target=True)
    y_test = normalize_data(y_test, target=True)

    return X_train, X_test, y_train, y_test


def test_only(X_test, y_test, weights_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(weights_file)
    print("Loaded model out of weights file successfully!")
    model.to(device)
    test_inputs = Variable(torch.FloatTensor(X_test)).to(device)
    test_labels = Variable(torch.FloatTensor(y_test))
    final_test_outputs = model(test_inputs)
    r2_score = metrics.r2_score(test_labels.detach().numpy(), final_test_outputs.detach().numpy())
    return r2_score * 100


def use_model(X_train, X_test, y_train, y_test, epochs, lr, h_units, opt_name, dropout, weights_file):
    if weights_file != "":
        test_predict = test_only(X_test, y_test, weights_file)
        return test_predict

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper Parameters
    batch_size = 512
    num_epochs = epochs
    learning_rate = lr
    size_hidden = h_units
    optimizer_name = opt_name
    p_dropout = dropout
    batch_no = len(X_train) // batch_size  # number of batches per epoch

    # Set model
    model = Model(X_train[0].shape[0], size_hidden, p_dropout)
    model.to(device)
    print(f"Model architecture is: {model}")
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("total trainable parameters: {}".format(pytorch_total_params))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_function = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Train loop
    running_loss = 0.0
    # Save data for plotting
    epoch_list = []
    loss_list = []
    test_loss_list = []
    r2_scores = []
    print("\nStarting Training:\n")
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

        test_labels = test_labels.detach().numpy()
        test_outputs = test_outputs.detach().numpy()
        r2_score = metrics.r2_score(test_labels, test_outputs)
        r2_scores.append(r2_score)

        # Print progress
        if epoch % 50 == 0:
            print('Epoch {}'.format(epoch), " | lr: ", np.round(scheduler.get_last_lr()[0], 6), " | loss:", np.round(running_loss/batch_no, 6), " | test loss: ", np.round(test_loss, 6), " | test R squared score: ", np.round(r2_score*100, 6), "%")
        if epoch % 250 == 0 and epoch > 0:
            scheduler.step()
        # Save data for plotting
        loss_list.append(running_loss/batch_no)
        test_loss_list.append(test_loss)
        epoch_list.append(epoch)
        running_loss = 0.0

    test_inputs = Variable(torch.FloatTensor(X_test))
    test_labels = Variable(torch.FloatTensor(y_test))
    test_inputs = test_inputs.to(device)
    final_test_outputs = model(test_inputs)
    test_labels = test_labels.detach().numpy()
    final_test_outputs = final_test_outputs.detach().numpy()
    print("\nSome values for example:\n")
    for i in range(10):
        print(f"The target truth value is {round(final_test_outputs[i][0]*target_std + target_mean)} while the predicted target is: {test_labels[i]*target_std + target_mean}")

    r2_score = metrics.r2_score(test_labels, final_test_outputs)

    # Print optimizer's state_dict
    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    torch.save(model.state_dict(), str("model_weights" + str(str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")) + ".pt"))

    # plt.plot(epoch_list, loss_list, label="train loss")
    # plt.plot(epoch_list, test_loss_list, label="test loss")
    # plt.xlabel("No. of epoch")
    # plt.ylabel("Loss")
    # plt.title(f"Epochs vs Loss\nModel: Hidden size: {size_hidden} | opt: {'SGD'} | lr: {learning_rate} | batch size: {batch_size} | Loss function: {'MSE'}")
    # plt.legend()
    # plt.savefig("Loss_VS_Epochs.png")

    data1 = np.asarray(final_test_outputs).squeeze()
    data1 = data1*target_std + target_mean
    data2 = np.asarray(test_labels)
    data2 = data2*target_std + target_mean
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff)

    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.title("Bland Altman plot ")
    plt.xlabel("Mean of predicted and truth values")
    plt.ylabel("Difference in prediction and truth values")
    plt.legend()
    plt.show()

    # plt.plot(epoch_list, r2_scores, label='r2 vs epochs')
    # plt.xlabel("No. of epoch")
    # plt.ylabel("R Squared score")
    # plt.title("Epochs vs R2 score")
    # plt.legend()
    # plt.savefig("R_Squared_VS_Epochs.png")
    return r2_score * 100


def dataset_dl_prediction(epochs, lr, h_units, opt_name, dropout, weights_file):
	# start deep learning model prediction
	print("\n")
	print('*' * 125)
	print("Starting Deep Learning algorithm for prediction of the 'RMR' feature")
	print('*' * 125)
	print("\n")

	# preprocess
	X_train, X_test, y_train, y_test = preprocess(DataSet.df)

	# train and test
	dl_prediction = use_model(X_train, X_test, y_train, y_test, epochs, lr, h_units, opt_name, dropout, weights_file)
	return dl_prediction
