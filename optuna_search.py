import optuna
import torch
from dl_model import preprocess, Model
from sklearn.utils import shuffle
from torch.autograd import Variable
from sklearn import metrics


def OptunaDefineModel(trial, input_dimensions):
    # n_layers = trial.suggest_int("n_layers", 1, 2)  # number of layers will be between 1 and 3
    layers = []
    n_layers = trial.suggest_int("n_layers", 1, 3)
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
    input_dimensions = X_train[0].shape[0]
    size_hidden = trial.suggest_int("n_units", 256, 1024)  # number of units will be between 16 and 2048
    drop = trial.suggest_float("dropout", 0, 0.5)  # dropout rate will be between 0 and 0.5

    # model = OptunaDefineModel(trial, input_dimensions).to(device)

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


def OptunaRunStudy(data_frame, epochs, n_trials, study_name):

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)
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
