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
    lr = 1e-5
    momentum = 0.2
    epochs = 400
    plot_training = True
    show_figs = False
    num_models = 5

    weights = [0.01, 0]
    epochs_arr = [10, 20, 50, 100, 150, 200, 500, 1000]

    losses = []
    acces = []
    f1es = []
    roces = []
    

    for epochs in epochs_arr:

        print(f"Learning Rate = {lr}, Epochs = {epochs}")
        dataset_losses = []
        dataset_accuracies = []
        dataset_f1_scores = []
        dataset_rocaucs = []

        train_loss = []
        train_acc = []
        train_f1 = []
        train_roc = []

        val_loss = []
        val_acc = []
        val_f1 = []
        val_roc = []
        
        for i in range(num_models):
            print(f"\rTraining model {i+1 : 2d}", end='')
            train_dl, val_dl, device, _, _ = get_dataloaders(x, y, batch_size, sample_size)
            model = Model(lr, epochs)
   
            model = model.to(device)

            t_loss, t_acc, t_f1, t_roc, v_loss, v_acc, v_f1, v_roc, epoch = model.fit(train_dl, val_dl)

            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_f1.append(t_f1)
            train_roc.append(t_roc)

            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_f1.append(v_f1)
            val_roc.append(v_roc) 

            dataset_x, _, dataset_y = preprocess_test(x, y)
            
            loss, accu, rocauc, f1_score = model.getMetrics(to_device(torch.tensor(dataset_x.to_numpy('float32')), device),
                                                  to_device(torch.tensor(dataset_y), device))
            
            dataset_losses.append(loss)
            dataset_accuracies.append(accu)
            dataset_f1_scores.append(f1_score)
            dataset_rocaucs.append(rocauc)
        print("")
        
        print(f"Average Loss: {np.mean(dataset_losses)}")
        print(f"Average Accuracy: {np.mean(dataset_accuracies)}")
        print(f"Average ROC-AUC: {np.mean(dataset_rocaucs)}")
        print(f"Average F1 Score: {np.mean(dataset_f1_scores)}")

        losses.append([np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)])
        acces.append([np.mean(train_acc, axis=0), np.mean(val_acc, axis=0)])
        f1es.append([np.mean(train_f1, axis=0), np.mean(val_f1, axis=0)])
        roces.append([np.mean(train_roc, axis=0), np.mean(val_roc, axis=0)])
        print("Done Training Models")
        print("")

    if plot_training:
        for i, epochs in enumerate(epochs_arr): 

            plt.title(f"Loss Vs Epochs, LR = {lr}, Epochs = {epochs}")
            x_range = range(1, epochs+1)
            plt.plot(x_range, losses[i][0], label="Training Loss")
            plt.plot(x_range, losses[i][1], label="Validation Loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f"Loss_V_Epochs_lr{lr}_Epochs_{epochs}.png")
            if show_figs:
                plt.show()

            plt.title(f"Accuracy Vs Epochs, LR = {lr}, Epochs = {epochs}") 
            plt.plot(x_range, acces[i][0], label="Training Accuracy")
            plt.plot(x_range, acces[i][1], label="Validation Accuracy")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.savefig(f"BalancedAccuracy_V_Epochs_lr{lr}_Epochs_{epochs}.png")
            if show_figs:
                plt.show()

            plt.title(f"F1 Score Vs Epochs, LR = {lr}, Epochs = {epochs}")
            plt.plot(x_range, f1es[i][0], label="Training F1")
            plt.plot(x_range, f1es[i][1], label="Validation F1")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.savefig(f"F1Score_V_Epochs_lr{lr}_Epochs_{epochs}.png")
            if show_figs:
                plt.show()
            
            plt.title(f"ROC-AUC Vs Epochs, LR = {lr}, Epochs = {epochs}")
            plt.plot(x_range, roces[i][0], label="Training ROC-AUC")
            plt.plot(x_range, roces[i][1], label="Validation ROC-AUC")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("ROC-AUC")
            plt.savefig(f"ROCAUC_V_Epochs_lr{lr}_Epochs_{epochs}.png")
            if show_figs:
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
