import torch
import torch.nn as nn 
import numpy as np 

from data import data,   X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, y_test_CV, y_train_CV

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

# 1. building a linear model. 1 input layer, 2 hidden layers and one output layer. ReLU activation function is used in between input and hidden.
n_features = X_train.shape[1] 
input_size = n_features
hidden_size = 100
hidden_size2 = 200
hidden_size3 = 100
output_size = 1 #return only a value 

class LinearModel(nn.Module):
    def __init__(self, n_inputs, hidden_size, hidden_size2, hidden_size3,n_outputs):
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
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 3. Training loops
num_epochs = 800

for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X_train_tensor)
    loss = criterion(y_predicted, y_train_CV)
    
    #clear gradient
    optimizer.zero_grad()
   
    #backward pass
    loss.backward()

    #update
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item()} ')

# 4. Model evaluation, r2 score and visualisation with one features
predicted = model(X_test_tensor).detach().numpy()
r2score = r2_score(y_test, predicted) #0.88 r2 value obtained. 
print(r2score)
# for MolLogP
plt.plot(X_test['MolLogP'], y_test, 'ro')
plt.plot(X_test['MolLogP'], predicted, 'b')
plt.show()
