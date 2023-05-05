# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from torch import optim


from data import *
from parser import parse
from model import Model


def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    
    batch_size = 128
    sample_size = 1000
    lr = 0.00005
    momentum = 0.2
    epochs = 50
    plot_training = False
    num_models = 5

    for lr in np.logspace(-5, -10, 6):

        print(f"Learning Rate = {lr : .10f}")
        losses = []
        accuracies = []
        f1_scores = []
        rocaucs = []
        
        for i in range(num_models):
            print(f"Training model {i+1 : 2d}", end="")
            train_dl, val_dl, device, _, _ = get_dataloaders(x, y, batch_size, sample_size)
            model = Model(lr, epochs)
   
            # model.optim_fn = optim.RMSprop(model.parameters(), model.lr, momentum=momentum)

            model = model.to(device)

            t_loss, t_acc, t_f1, v_loss, v_acc, v_f1, epoch = model.fit(train_dl, val_dl)

            print(f"  Model ran for {epoch} epochs")
            

            dataset_x, _, dataset_y = preprocess_test(x, y)
            
            loss, accu, rocauc, f1_score = model.getMetrics(to_device(torch.tensor(dataset_x.to_numpy('float32')), device),
                                                  to_device(torch.tensor(dataset_y), device))
            
            losses.append(loss)
            accuracies.append(accu)
            f1_scores.append(f1_score)
            rocaucs.append(rocauc)

        print("")

        print(f"Average Loss: {np.mean(losses)}")
        print(f"Average Accuracy: {np.mean(accuracies)}")
        print(f"Average ROC-AUC: {np.mean(rocaucs)}")
        print(f"Average F1 Score: {np.mean(f1_scores)}")
        print("")

        if plot_training:
            plt.title(f"Loss Vs Epochs, LR = {lr}")
            x_range = range(1, epochs+1)
            plt.plot(x_range, t_loss, label="Training Loss")
            plt.plot(x_range, v_loss, label="Validation Loss")

            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

            plt.show()

            plt.title(f"Accuracy Vs Epochs, LR = {lr}") 
            plt.plot(x_range, t_acc, label="Training Accuracy")
            plt.plot(x_range, v_acc, label="Validation Accuracy")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.show()

            plt.title(f"F1 Score Vs Epochs, LR = {lr}")
            plt.plot(x_range, t_f1, label="Training F1")
            plt.plot(x_range, v_f1, label="Validation F1")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.show()

    # test_x = load_data("test_x.csv")

    # x_test, ids = preprocess_test(test_x)

    # probs = model.predict_proba(to_device(torch.tensor(x_test.to_numpy('float32')), device))

    # submission = list(zip(ids, probs))

    # with open("submission.csv", "w") as file:
    #     file.write("patientunitstayid,hospitaldischargestatus \r\n")
    #     for id, prob in submission:
    #         file.write(f"{int(id)},{prob :f}\r\n")

if __name__ == "__main__":
    main()
