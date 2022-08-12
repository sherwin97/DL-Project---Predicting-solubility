import argparse

import numpy as np 
import pandas as pd 

from sklearn.metrics import r2_score

import torch 
import torch.nn as nn


def to_tensor(X_train, X_test, y_train, y_test):
    #load csv files
    X_train = pd.read_csv(X_train)
    X_test = pd.read_csv(X_test)
    y_train = pd.read_csv(y_train)
    y_test = pd.read_csv(y_test)

    # converting df to tensor
    X_train_tensor = torch.from_numpy(X_train.to_numpy().astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test.to_numpy().astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train.to_numpy().astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.to_numpy().astype(np.float32))

    # rehshaping to column vector
    y_train_CV = y_train_tensor.view(y_train_tensor.shape[0], 1)
    y_test_CV = y_test_tensor.view(y_test_tensor.shape[0], 1)

    return X_train_tensor, X_test_tensor, y_train_CV, y_test_CV

def training_model(X_train, X_test, y_train, y_test, saved_model):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = to_tensor(X_train, X_test, y_train, y_test)
    '''
    Creates a neural network which 3 hidden layers
    '''

    #defining variables for neural network
    n_features = X_train_tensor.shape[1]
    input_size = n_features
    hidden_size = 100
    hidden_size2 = 200
    hidden_size3 = 100
    output_size = 1  # return only a value

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


    model = LinearModel(input_size, hidden_size, hidden_size2, hidden_size3, output_size)

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

    torch.save(model.state_dict(), saved_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_train", help="Enter the file path to load X_train")
    parser.add_argument("--X_test", help="Enter the file path to load X_test")
    parser.add_argument("--y_train", help="Enter the file path to load y_train")
    parser.add_argument("--y_test", help="Enter the file path to load y_test")
    parser.add_argument("--model", help="Enter the file path to save model") 
    args = parser.parse_args()

    training_model(args.X_train, args.X_test, args.y_train, args.y_test, args.model)