import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn


def load_data(path_features, path_solubility, test_size, random_state):
    """
    load in csv files containing features and solubility. split data according to user defined sizes. convert df to tensor. 
    
    """
    # load csv files
    features_df = pd.read_csv(path_features)
    solubility_df = pd.read_csv(path_solubility)

    # splitting, user to specify the test size and random state else by default, test size=0.2 and random_state=123
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, solubility_df, test_size=test_size, random_state=random_state
    )

    # normalising
    X_train = X_train / 100
    X_test = X_test / 100
    y_train = y_train / 100
    y_test = y_test / 100

    # converting df to tensor
    X_train_tensor = torch.from_numpy(X_train.to_numpy().astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test.to_numpy().astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.to_numpy().astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.float32))

    # rehshaping to column vector
    y_train_CV = y_train_tensor.view(y_train_tensor.shape[0], 1)
    y_test_CV = y_test_tensor.view(y_test_tensor.shape[0], 1)

    return X_train_tensor, X_test_tensor, y_train_CV, y_test_CV


def training_model(
    path_features,
    path_solubility,
    test_size,
    random_state,
    n_features,
    num_hidden_layers,
    hidden_layer_size,
    path_to_trained_model,
):

    """
    Creates a neural network containing user defined variables. 
    """
    # load data
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = load_data(
        path_features, path_solubility, test_size, random_state
    )

    class LinearModel(nn.Module):
        def __init__(self, n_features, num_hidden_layers, hidden_layer_size):
            output_size = 1
            super(LinearModel, self).__init__()
            self.module = nn.Sequential()
            for i in range(num_hidden_layers):
                self.module.add_module(
                    f"ff{i}", nn.Linear(n_features, hidden_layer_size)
                )
                self.module.add_module(f"activation{i}", nn.ReLU())
                n_features = hidden_layer_size
            self.module.add_module(
                f"last layer", nn.Linear(hidden_layer_size, output_size)
            )

        def forward(self, x):
            out = self.module(x)
            return out

    model = LinearModel(n_features, num_hidden_layers, hidden_layer_size)

    # 2. Loss and optimizer
    learning_rate = 0.01

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training loops
    num_epochs = 800
    for epoch in range(num_epochs):
        # forward pass and loss
        y_predicted = model(X_train_tensor)
        loss = criterion(y_predicted, y_train_tensor)

        # clear gradient
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"epoch: {epoch+1}, loss = {loss.item()} ")

    predicted = model(X_test_tensor).detach().numpy()
    r2score = r2_score(y_test_tensor, predicted)
    print(r2score)
    torch.save(model.state_dict(), path_to_trained_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_features", help="Enter the file path to load features csv"
    )
    parser.add_argument(
        "--path_solubility", help="Enter the file path to load solubility csv"
    )
    parser.add_argument(
        "--test_size", help="Enter the test size", type=float, default=0.2
    )
    parser.add_argument(
        "--random_state",
        help="Enter an integer to fix random seed",
        type=int,
        default=123,
    )
    parser.add_argument("--n_features", help="Enter the number of features", type=int)
    parser.add_argument(
        "--num_hidden_layers", help="Enter the file path to save model", type=int
    )
    parser.add_argument(
        "--hidden_layer_size", help="Enter the file path to save model", type=int
    )
    parser.add_argument(
        "--path_to_trained_model", help="Enter the file path to save model"
    )

    args = parser.parse_args()

    training_model(
        args.path_features,
        args.path_solubility,
        args.test_size,
        args.random_state,
        args.n_features,
        args.num_hidden_layers,
        args.hidden_layer_size,
        args.path_to_trained_model,
    )
