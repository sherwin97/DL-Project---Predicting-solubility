import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def train_test_data(features, y, output1, output2, output3, output4):
    features = pd.read_csv(features)
    y = pd.read_csv(y)

    # splitting
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=123
    )

    # normalising
    X_train = X_train / 100
    X_test = X_test / 100
    y_train = y_train / 100
    y_test = y_test / 100

    return (
        X_train.to_csv(output1, index = False),
        X_test.to_csv(output2, index = False),
        y_train.to_csv(output3, index = False),
        y_test.to_csv(output4, index=False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="Enter the file path containing features")
    parser.add_argument("--y", help="Enter the file path containing y")
    parser.add_argument("--X_train", help="Enter the file path to save X_train")
    parser.add_argument("--X_test", help="Enter the file path to save X_test")
    parser.add_argument("--y_train", help="Enter the file path to save y_train")
    parser.add_argument("--y_test", help="Enter the file path to save y_test")
    
    args = parser.parse_args()

    train_test_data(args.features, args.y, args.X_train, args.X_test, args.y_train, args.y_test)
   

