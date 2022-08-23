import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, n_features, num_hidden_layers, hidden_layer_size):
        output_size = 1
        super(LinearModel, self).__init__()
        self.module = nn.Sequential()
        for i in range(num_hidden_layers):
            self.module.add_module(f"ff{i}", nn.Linear(n_features, hidden_layer_size))
            self.module.add_module(f"activation{i}", nn.ReLU())
            n_features = hidden_layer_size
        self.module.add_module(f"last layer", nn.Linear(hidden_layer_size, output_size))

    def forward(self, x):
        out = self.module(x)
        return out


def predict(
    path_features,
    n_features,
    num_hidden_layers,
    hidden_layer_size,
    path_to_trained_model,
    path_to_pred_sol,
):
    """
    Load in features data, y data and saved model from previous training. Predict the data and save as a csv. Also return r value if y_actual is included. 
    """
    # load features
    features = pd.read_csv(path_features)

    # normalise data
    features = features / 100

    # converting df to tensor
    X_tensor = torch.from_numpy(features.to_numpy().astype(np.float32))

    # initialise model, load model
    model = LinearModel(n_features, num_hidden_layers, hidden_layer_size)
    model.load_state_dict(torch.load(path_to_trained_model))
    pred = model(X_tensor).detach().numpy()
    predicted_unscaled = pred * 100
    df_predicted = pd.DataFrame(predicted_unscaled)
    return df_predicted.to_csv(path_to_pred_sol, index=False, header="Solubility")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_features", help="Enter the file path to load features csv"
    )
    parser.add_argument("--n_features", help="Enter the number of features", type=int)
    parser.add_argument(
        "--num_hidden_layers", help="Enter the number of hidden layers", type=int
    )
    parser.add_argument(
        "--hidden_layer_size", help="Enter the size of hidden layer", type=int
    )
    parser.add_argument(
        "--path_to_trained_model", help="Enter path to load trained linear model"
    )
    parser.add_argument(
        "--path_to_pred_sol", help="Enter path to save predicted solubilities"
    )

    args = parser.parse_args()

    predict(
        args.path_features,
        args.n_features,
        args.num_hidden_layers,
        args.hidden_layer_size,
        args.path_to_trained_model,
        args.path_to_pred_sol,
    )
