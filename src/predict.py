import pandas as pd
import numpy as np

from src.config import config
import src.config.preprocessing.preprocessors as pp
from src.config.preprocessing.data_management import load_dataset, save_model, load_model

import pipeline as pl
import traine_pipeline as tt

import pickle

X_test = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})
Y_test = [0, 1, 1, 0]  # Replace with your actual test labels

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(sample):
    h = {}
    h[0] = sample.values.reshape(1, sample.shape[0])

    for l in range(1, config.NUM_LAYERS):
        z = tt.layer_neurons_weighted_sum(h[l-1], pl.theta0[l], pl.theta[l])
        h[l] = tt.layer_neurons_output(z, config.f[l])

    final_output = sigmoid(h[config.NUM_LAYERS-1])[0, 0]

    binary_output = 1 if final_output >= 0.8 else 0  # Adjust threshold to 0.8

    return binary_output, final_output

if __name__ == "__main__":
    # Load and preprocess your training data
    training_data = load_dataset("train.csv")
    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    # Initialize parameters
    pl.initialize_parameters()

    correct_predictions = 0

    # Evaluate on X_test
    for sample_index in range(len(X_test)):
        sample = X_test.iloc[sample_index]
        binary_output, final_output = forward_pass(sample)
        print(f"Binary output for sample {sample_index}: {binary_output} (final_output: {final_output:.4f})")

        # Compare with ground truth
        if binary_output == Y_test[sample_index]:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(X_test) * 100
    print(f"\nAccuracy on test set: {accuracy:.2f}%")
