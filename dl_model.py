import datetime
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
import optuna


class Model(torch.nn.Module):
    def __init__(self, input_dimensions, hidden_size, dropout):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dimensions, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.shape_outputs = torch.nn.Linear(hidden_size, 1)
        # TODO: check with and without this initialization
        # torch.nn.init.xavier_uniform_(self.shape_outputs.weight)

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


def train(X_train, X_test, y_train, y_test, epochs, lr, h_units, opt_name, dropout):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper Parameters
    batch_size = len(X_train)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

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
            print('Epoch {}'.format(epoch), " | lr: ", np.round(scheduler.get_last_lr()[0], 6), " | loss:", np.round(running_loss, 6), " | test loss: ", np.round(test_loss, 6), " | test R squared score: ", np.round(r2_score*100, 6), "%")
        if epoch % 500 == 0 and 2501 > epoch > 0:
            scheduler.step()
        # Save data for plotting
        loss_list.append(running_loss)
        test_loss_list.append(test_loss)
        epoch_list.append(epoch)
        running_loss = 0.0

    test_inputs = Variable(torch.FloatTensor(X_test))
    test_labels = Variable(torch.FloatTensor(y_test))
    test_inputs = test_inputs.to(device)
    final_test_outputs = model(test_inputs)
    test_labels = test_labels.detach().numpy()
    final_test_outputs = final_test_outputs.detach().numpy()
    r2_score = metrics.r2_score(test_labels, final_test_outputs)

    # Print optimizer's state_dict
    print("\nOptimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    torch.save(model.state_dict(), str("model_weights" + str(str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")) + ".pt"))

    plt.plot(epoch_list, loss_list, label="train loss")
    plt.plot(epoch_list, test_loss_list, label="test loss")
    plt.xlabel("No. of epoch")
    plt.ylabel("Loss")
    plt.title(f"Epochs vs Loss\nModel: Hidden size: {size_hidden} | opt: {'SGD'} | lr: {learning_rate} | batch size: {batch_size} | Loss function: {'MSE'}")
    plt.legend()
    plt.show()

    plt.plot(epoch_list, r2_scores, label='r2 vs epochs')
    plt.xlabel("No. of epoch")
    plt.ylabel("R Squared score")
    plt.title("Epochs vs R2 score")
    plt.legend()
    plt.show()
    return r2_score * 100


def OptunaDefineModel(trial, input_dimensions):
    # n_layers = trial.suggest_int("n_layers", 1, 2)  # number of layers will be between 1 and 3
    layers = []
    n_layers = 1
    in_features = input_dimensions
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 2048)  # number of units will be between 16 and 2048
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0, 0.5)  # dropout rate will be between 0 and 0.5
        layers.append(torch.nn.Dropout(p))
        in_features = out_features

    layers.append(torch.nn.Linear(in_features, 1))
    layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


def OptunaTrainObjective(trial, df, epochs):
    X_train, X_test, y_train, y_test = preprocess(df)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate the model.
    # input_dimensions = X_train[0].shape[0]
    # model = define_model(trial, input_dimensions).to(device)
    size_hidden = trial.suggest_int("n_units", 256, 8192)  # number of units will be between 16 and 2048
    drop = trial.suggest_float("dropout", 0, 0.5)  # dropout rate will be between 0 and 0.5

    model = Model(X_train[0].shape[0], size_hidden, drop).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)  # log=True, will use log scale to interpolate between lr
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    batch_size = len(X_train)
    epochs = epochs
    loss_function = torch.nn.MSELoss()
    batch_no = len(X_train) // batch_size  # number of batches per epoch
    running_loss = 0
    # Training of the model.
    for epoch in range(epochs):
        model.train()
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
        with torch.no_grad():
            model.eval()
            test_inputs = Variable(torch.FloatTensor(X_test))
            test_labels = Variable(torch.FloatTensor(y_test))
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_inputs)

            test_labels = test_labels.detach().numpy()
            test_outputs = test_outputs.detach().numpy()
            r2_score = metrics.r2_score(test_labels, test_outputs)

        running_loss = 0.0

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(r2_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return r2_score


def OptunaRunStudy(data_frame, epochs, n_trials):

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="RMR-fc", direction="maximize", sampler=sampler)
    study.optimize(lambda trial: OptunaTrainObjective(trial, data_frame, epochs), n_trials=n_trials, timeout=1800)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study).write_image("param_importance.png")
    optuna.visualization.plot_optimization_history(study).write_image("optimization_history.png")
    optuna.visualization.plot_intermediate_values(study).write_image("intermediate_valuse.png")

    optuna.visualization.plot_contour(study, params=["n_units", "dropout"]).write_image("n_units_vs_dropout.png")
    optuna.visualization.plot_contour(study, params=["n_units", "lr"]).write_image("n_units_vs_lr.png")
    optuna.visualization.plot_contour(study, params=["lr", "dropout"]).write_image("lr_vs_dropout.png")
