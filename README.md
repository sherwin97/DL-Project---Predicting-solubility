# DL-Project---Predicting-solubility
A project on using Pytorch to train a Neural Network to predict the solubility of molecules using its molecular descriptors. 

Data obtained from:
John S. Delaney
Journal of Chemical Information and Computer Sciences 2004 44 (3), 1000-1005
DOI: 10.1021/ci034243x

To use run the codes, one may clone the repository using the following command in the terminal

```
git clone https://github.com/sherwin97/DL-Project---Predicting-solubility.git
```

This repository consists of 4 folders and 1 txt file. 

1. data_csv folder contains the csv files. 
   1. ci034243xsi20040112_053635.csv contains the original file
   2. smiles.csv contains the SMILES data 
   3. solubility.csv contains the solubility data
   4. features.csv contains the features data extracted using SMILES and RDKit
   5. pred_sol.csv contains the predicted solubility using the trained neural model 

2. notebooks folder contains jupyter notebook files for testing. 

3. python_scripts folder contains three python scripts to obtain features, train model and predict. 
    1. process_data.py takes in the path of SMILES data and return a csv of features into the data_csv folder. 
    To run the code, one may type the following into the terminal:
    ```
    python ./python_scripts/process_data.py --path_smiles ./data_csv/smiles.csv --path_features ./data_csv/features.csv
    ```
    2. training_model.py takes in the path of features and solubility. Then user may define their test size and random state for seeding. Else the default would be 0.2 for test size and 123 for random state. Next, user may specify the number of features, number of hidden layers, size of hidden layer, and the path to save the trained model. 
    To run the code, one may type the following into the terminal:
    ```
    python ./python_scripts/training_model.py --path_features ./data_csv/features.csv --path_solubility ./data_csv/solubility.csv --test_size 0.2 --random_state 123 --n_features 5 --num_hidden layers 3 --hidden_layer_size 20 --path_to_trained_model ./trained_model/trained_linear_model.pt
    ```

    3. prediction.py allow user to load in the trained model and the predict the solubility of specific molecules given its molecular descriptors. Do note that the n_features, num_hidden_layers and size_hidden_layers should be the same as the trained model. A csv of predicted solubility will be returned in the data_csv folder. 
    To run the code, one may type the following into the terminal. 
    ```
    python ./python_scripts/prediction.py --path_features ./data_csv/features.csv --n_features 5 --num_hidden_layers 3 --size_hidden_layer 20 --path_to_trained_model ./trained_model/trained_linear_model.pt -- ./data_csv/pred_sol.csv
    ```

4. requirements.txt specify the version information for the codes to function properly. 
