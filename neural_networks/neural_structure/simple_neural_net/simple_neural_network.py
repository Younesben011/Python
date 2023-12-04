from neuron import neuron
import pandas as pd
import os.path as path
import numpy as np


DATASET_PATH = "G:\\coding\\Python\\neural_networks\\neural_structure\\simple_neural_net\\datasets"
FILE_NAME = "dataset2.csv"


def Load_dataset():
    file = path.join(DATASET_PATH, FILE_NAME)
    data = pd.read_csv(file, index_col=False)
    df = pd.DataFrame(data)
    outputs = df.pop('y').to_numpy()
    inputs = df.to_numpy()
    return (inputs, outputs)


def traning(inputs, outputs, N):

    for input, output in zip(inputs, outputs):
        calculated_output, weights = N.predict(input)
        Err = output-calculated_output
        if (Err != 0):
            N.update_weights(Err)
    return (weights)


def calculate_accuracy(inputs, outputs, N):
    best_acc = 0
    best_weights = []
    all_weights = []
    accuracy = 0
    all_accuracies = []
    predicted_lsit = []
    for input in inputs:
        calculated_output, weights = N.predict(input)
        predicted_lsit.append(calculated_output)
        all_weights.append(weights)

    accuracy = np.sum(predicted_lsit == outputs)/len(outputs)
    if (accuracy == 1):
        return (1, weights,)
    elif (accuracy > best_acc):
        best_weights = weights
    return (accuracy, best_weights)


def main():
    repeat = 0
    acc = 0
    inputs, outputs = Load_dataset()
    inputs_number = len(inputs[0])+1
    all_weights = []
    all_accuracies = []
    N = neuron(inputs_number=inputs_number, nu=0.9)
    while (repeat < 4 and acc != 1):
        repeat += 1
        all_weights.append(traning(inputs, outputs, N))

        acc, final_weights = calculate_accuracy(
            inputs, outputs, N)
        all_accuracies.append(acc)
    print(20*"=", "result", 20*"=")
    print(
        f"the best possible weights are => {final_weights} \nwith accuracy of => {acc*100}%\nattempt {repeat}")
    details = input("see details (Y/N)=> ").capitalize() == "Y"
    if (details):
        print(20*"=", "details", 20*"=")
        attempt = 1
        for weights, accuracy in zip(all_weights, all_accuracies):
            print(
                f"attempt number({attempt}):\nweights => {weights} \nwith accuracy of => {round(accuracy,2)*100}%")
            attempt += 1


main()
