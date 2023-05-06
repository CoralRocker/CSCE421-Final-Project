
import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from torch import optim


from data import *
from parser import parse
from model import Model

np.seterr(all='raise')

def main():
    args = parse()

    print(f"Default device: {get_default_device()}")
    
    device = get_default_device()

    print("Loading X and Y")
    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    
    batch_size = 128
    sample_size = 6000
    lr = 1e-5
    epochs = 100
    hidden_size = 100
    num_layers = 3

    num_models = 5


    print("Generating Test Dataset")
    dataset_x, _, dataset_y = preprocess_test(x, y)
    dataset_x = to_device(torch.tensor(dataset_x.to_numpy('float32')), device)
    dataset_y = to_device(torch.tensor(dataset_y), device)

    model_names = ["Oversampled Thin NN", "Oversampled Dense NN",  "Raw Sample Thin NN", "Raw Sample Dense NN"] 
    model_accuracies = []
    ypos = np.arange(len(model_names))

    ### Oversampled
    print("Generating Dataloaders")
    train_dl, val_dl, device, _, _ = get_dataloaders(x, y, batch_size, sample_size)
    # Dense
    model = Model(lr, epochs, hidden_size, num_layers) 
    model = model.to(device)
    model.fit(train_dl, val_dl)
    loss, accu, rocauc, f1_score = model.getMetrics(dataset_x, dataset_y)
    model_accuracies.append(rocauc)
    
    # Thin
    model = Model(lr, epochs, 49, 0) 
    model = model.to(device)
    model.fit(train_dl, val_dl)
    loss, accu, rocauc, f1_score = model.getMetrics(dataset_x, dataset_y)
    model_accuracies.append(rocauc)


    ### Unsampled
    print("Generating Dataloaders")
    train_dl, val_dl, device, _, _ = get_dataloaders(x, y, batch_size, 0)

    # Dense
    model = Model(lr, epochs, hidden_size, num_layers) 
    model = model.to(device)
    model.fit(train_dl, val_dl)
    loss, accu, rocauc, f1_score = model.getMetrics(dataset_x, dataset_y) 
    model_accuracies.append(rocauc)

    # Thin
    model = Model(lr, epochs, 49, 0) 
    model = model.to(device)
    model.fit(train_dl, val_dl)
    loss, accu, rocauc, f1_score = model.getMetrics(dataset_x, dataset_y) 
    model_accuracies.append(rocauc)

    ### Plot Bar Chart
    plt.barh(ypos, model_accuracies)
    plt.yticks(ypos[::-1], labels=model_names[::-1])
    plt.xlabel("AUC-ROC")
    plt.title("AUC-ROC Vs. Sampling Method")
    plt.show()


if __name__ == "__main__":
    main()
