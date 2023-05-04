# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


from data import *
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    
    batch_size = 128
    sample_size = 3000

    train_dl, val_dl, device = get_dataloaders(x, y, batch_size, sample_size)

    lr = 0.0005
    epochs = 50

    model = Model(lr, epochs)
    model = model.to(device)

    t_loss, t_acc, v_loss, v_acc = model.fit(train_dl, val_dl)

    if False:
        plt.title("Loss Vs Epochs")
        x_range = range(1, epochs+1)
        plt.plot(x_range, t_loss, label="Training Loss")
        plt.plot(x_range, v_loss, label="Validation Loss")

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.show()

        plt.title("Accuracy Vs Epochs") 
        plt.plot(x_range, t_acc, label="Training Accuracy")
        plt.plot(x_range, v_acc, label="Validation Accuracy")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    test_x = load_data("test_x.csv")
    x_test = preprocess_x(test_x)

    ids = x_test['patientunitstayid'].to_numpy()
    x_test.drop('patientunitstayid', axis=1, inplace=True)


    probs = model.predict_proba(to_device(torch.tensor(x_test.to_numpy('float32')), get_default_device()))

    submission = list(zip(ids, probs))

    with open("submission.csv", "w") as file:
        file.write("patientunitstayid, hospitaldischargestatus\n")
        for id, prob in submission:
            file.write(f"{int(id)}, {prob :f}\n")

    # train_x, train_y, test_x, test_y = split_data(x, y)

    # ###### Your Code Here #######
    # # Add anything you want here
    # ############################

    # processed_x_train = preprocess_x(train_x)
    # processed_x_test = preprocess_x(test_x)

    # ###### Your Code Here #######
    # # Add anything you want here
    # ############################

    # model = Model(args)  # you can add arguments as needed
    # model.fit(processed_x_train, train_y)
    # x = load_data("test_x.csv")

    # ###### Your Code Here #######
    # # Add anything you want here

    # ############################

    # processed_x_test = preprocess_x(x)

    # prediction_probs = model.predict_proba(processed_x_test)

    #### Your Code Here ####
    # Save your results

    ########################


if __name__ == "__main__":
    main()
