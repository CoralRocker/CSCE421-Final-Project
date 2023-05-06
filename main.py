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

np.seterr(all='raise')

def main():
    args = parse()

    x = load_data("train_x.csv")
    y = load_data("train_y.csv")
    
    batch_size = 128
    sample_size = 6000
    lr = 1e-5
    momentum = 0.2
    epochs = 150
    plot_training = True
    show_figs = False
    num_models = 5

    epochs_arr = [75, 100, 125, 150]
    sample_size_arr = [2000, 4000, 6000]

    hidden_layer_size = [25, 50, 75, 100]
    hidden_layers = [0]

    losses = []
    acces = []
    f1es = []
    roces = []
    
    
    
    for num_layers in hidden_layers:
        
        layer_loss = []
        layer_acc = []
        layer_f1 = []
        layer_roc = []
        

        for hidden_size in hidden_layer_size:

            print(f"Learning Rate = {lr}, Epochs = {epochs}, Hidden Layer Size = {hidden_size}, {num_layers} Hidden Layers")
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
                model = Model(lr, epochs, hidden_size, num_layers)
   
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

            layer_loss.append([np.mean(train_loss, axis=0), np.mean(val_loss, axis=0)])
            layer_acc.append([np.mean(train_acc, axis=0), np.mean(val_acc, axis=0)])
            layer_f1.append([np.mean(train_f1, axis=0), np.mean(val_f1, axis=0)])
            layer_roc.append([np.mean(train_roc, axis=0), np.mean(val_roc, axis=0)])
            print("Done Training Models")
            print("")

        losses.append(layer_loss)
        acces.append(layer_acc)
        f1es.append(layer_f1)
        roces.append(layer_roc)

    if plot_training:
        for j, num_layers in enumerate(hidden_layers):
            for i, layer_size in enumerate(hidden_layer_size): 
                file_desciptor = f"LR_{lr}__Epochs_{epochs}__NumHiddenLayers_{num_layers}__LayerSize_{layer_size}"
                graph_title_suffix = f"LR = {lr}, Epochs = {epochs}, # Hidden Layers = {num_layers}, Hidden Layer Size = {layer_size}"

                plt.title(f"Loss Vs Epochs, {graph_title_suffix}")
                x_range = range(1, epochs+1)
                plt.plot(x_range, losses[j][i][0], label="Training Loss")
                plt.plot(x_range, losses[j][i][1], label="Validation Loss")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(f"Loss_V_Epochs_{file_desciptor}.png")
                if show_figs:
                    plt.show()
                else:
                    plt.clf()

                plt.title(f"Accuracy Vs Epochs, {graph_title_suffix}") 
                plt.plot(x_range, acces[j][i][0], label="Training Accuracy")
                plt.plot(x_range, acces[j][i][1], label="Validation Accuracy")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.savefig(f"BalancedAccuracy_V_Epochs_{file_desciptor}.png")
                if show_figs:
                    plt.show()
                else:
                    plt.clf()

                plt.title(f"F1 Score Vs Epochs, {graph_title_suffix}")
                plt.plot(x_range, f1es[j][i][0], label="Training F1")
                plt.plot(x_range, f1es[j][i][1], label="Validation F1")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("F1 Score")
                plt.savefig(f"F1Score_V_Epochs_{file_desciptor}.png")
                if show_figs:
                    plt.show()
                else:
                    plt.clf()
                
                plt.title(f"ROC-AUC Vs Epochs, {graph_title_suffix}")
                plt.plot(x_range, roces[j][i][0], label="Training ROC-AUC")
                plt.plot(x_range, roces[j][i][1], label="Validation ROC-AUC")
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("ROC-AUC")
                plt.savefig(f"ROCAUC_V_Epochs_{file_desciptor}.png")
                if show_figs:
                    plt.show()
                else:
                    plt.clf()




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
