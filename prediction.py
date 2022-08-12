import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

import torch
import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, n_inputs, hidden_size, hidden_size2, hidden_size3, n_outputs):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, n_outputs)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)

        return out


def predict(model_path, features_path, actual_y_path, predicted_y_path):
    '''
    Load in features data, y data and saved model from previous training. Predict the data and save as a csv. Also return r value if y_actual is included. 
    '''
    features = pd.read_csv(features_path)
    actual_y = pd.read_csv(actual_y_path)

    input_size = 5
    hidden_size = 100
    hidden_size2 = 200
    hidden_size3 = 100
    output_size = 1  # return only a value

    # converting df to tensor
    X_tensor = torch.from_numpy(features.to_numpy().astype(np.float32))
    y_tensor = torch.from_numpy(actual_y.to_numpy().astype(np.float32))

    # rehshaping to column vector
    y_tensor_CV = y_tensor.view(y_tensor.shape[0], 1)
    model = LinearModel(input_size, hidden_size, hidden_size2, hidden_size3, output_size)
    model.load_state_dict(torch.load(model_path))
    predicted = model(X_tensor).detach().numpy()
    r2score = r2_score(actual_y, predicted)
    predicted_unscaled = np.multiply(predicted, 100)  
    print(r2score)
    df_predicted = pd.DataFrame(predicted_unscaled)
    return df_predicted.to_csv(predicted_y_path, index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Enter the file path to load X_train")
    parser.add_argument("--features", help="Enter the file path to load X_test")
    parser.add_argument("--actual_y", help="Enter the file path to load y_train")
    parser.add_argument("--predicted_y", help="Enter path to save predicted y values")

    args = parser.parse_args()

    predict(args.model, args.features, args.actual_y, args.predicted_y)

