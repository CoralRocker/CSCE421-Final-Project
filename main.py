# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score


from data import *
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")

    train_dl, val_dl, device = get_dataloaders(x, y)

    lr = 0.001
    epochs = 100

    model = Model(lr, epochs)
    model = model.to(device)

    model.fit(train_dl, val_dl)

    # test_x = load_data("test_x.csv")
    # x_test = preprocess_x(test_x)

    # probs = model.predict_proba(to_device(torch.tensor(x_test.to_numpy('float32')), get_default_device()))
    # print(probs)

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
