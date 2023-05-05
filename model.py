from torch.nn import *
from torch import optim
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import matplotlib.pyplot as plt

from data import get_default_device, to_device

## Class for early stopping training. Based on an answer from the following thread
##  https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
##
class EarlyTrainingStop():

    ## Create an Early Training Stopper
    ## Tolerance is how many times the loss may increase before being halted 
    ## Delta is how much the validation loss may deviate from the training loss without being counted as significant
    def __init__(self, tolerance=1, delta=0):
        self.tolerance = tolerance
        self.delta = delta
        self.count = 0

    ## Check whether the stopping criterion is met
    ##
    ## Return true to stop, false to continue
    def __call__(self, train_loss, val_loss):
        if (val_loss - train_loss) > self.delta:
            self.count += 1
            if self.count > self.tolerance:
                return True
        else:
            self.count = 0

        return False

class Model(Module):
    def __init__(self, lr=0.001, n_epochs=5):
        super(Model, self).__init__()

        self.lr = lr
        self.epochs = n_epochs


        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        ########################################################################

        self.layer = Sequential(
                Linear(11, 30),
                Tanh(),
                Linear(30, 15),
                ReLU(),
                Linear(15, 2),
                Softmax(1),
                # Linear(15, 1),
                # Sigmoid(),
                )


        
        self.optim_fn = optim.Adam(self.parameters(), lr=self.lr)
        class_weights = to_device(torch.Tensor([1848 / 2016, 168 / 2016]), get_default_device())
        self.loss_fn = CrossEntropyLoss(class_weights) 

    def forward(self, X):
        return self.layer(X)


    def fit(self, train_dl, val_dl):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        ########################################################################

        train_accuracies = []
        train_losses = []
        train_f1 = []
        validation_accuracies = []
        validation_losses = []
        validation_f1 = []
       
        epoch = 0
        
        for epoch in range(self.epochs):
            
            earlystop = EarlyTrainingStop(1, 0.005)

            val_accuracy = None
            train_accuracy = None
            total = 0
            correct = 0

            tr_losses = []
            tr_f1 = []
            val_losses = []
            val_f1 = []
            train_loss = 0
            val_loss = 0

            for i, (x, y) in enumerate(train_dl):
                pred = self(x)

                self.optim_fn.zero_grad()

                loss = self.loss_fn(pred, y)

                loss.backward()

                self.optim_fn.step()
                
                tr_losses.append(loss.item())
                _, classes = torch.max(pred, dim=1)
                correct += torch.sum(classes == y).item()
                
                #correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += pred.size(0)
                
                tr_f1.append(f1_score(y.cpu().detach(), classes.cpu().detach(), zero_division=0))

                # if epoch == self.epochs-1 and i == 0:
                # #     prec, rec, thresh = precision_recall_curve(y.cpu().detach(), pred.squeeze().cpu().detach())
                # #     plt.title("Precision Recall Curve")
                # #     plt.plot(rec, prec)
                # #     plt.xlabel("Recall")
                # #     plt.ylabel("Precision")
                # #     plt.show()
                #     print(f"AUC-ROC Score: {roc_auc_score(y.cpu().detach(), pred.squeeze().cpu().detach()) :f}")


            train_accuracy = correct / total
            train_loss = np.mean(tr_losses)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_f1.append(np.mean(tr_f1))

            total = 0
            correct = 0

            for (x, y) in val_dl:
                pred = self.forward(x)

                loss = self.loss_fn(pred, y)
                val_losses.append(loss.item())

                self.optim_fn.step()

                _, classes = torch.max(pred, dim=1)

                val_f1.append(f1_score(y.cpu().detach(), classes.cpu().detach(), zero_division=0))
                correct += torch.sum(classes == y).item()
                # correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += x.size(0)


            val_loss = np.mean(val_losses)
            val_accuracy = correct / total
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            validation_f1.append(np.mean(val_f1))

            if earlystop(train_loss, val_loss):
                print(f"Stopping early at epoch {epoch+1}")
                break;

            # if (epoch + 1) % 5 == 0:  
            #     # Print progress
            #     if val_accuracy is not None:
            #         print("Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_accuracy: {:.4f}, val_accuracy: {:.4f}"
            #                   .format(epoch+1, self.epochs, train_loss, val_loss, train_accuracy, val_accuracy))
            #     else:
            #         print("Epoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.4f}"
            #                   .format(epoch+1, self.epochs, train_loss, train_accuracy))

        return train_losses, train_accuracies, train_f1, validation_losses, validation_accuracies, validation_f1, epoch+1
            

    def getMetrics(self, X, Y):

        total = 0
        correct = 0
        losses = []

        pred = self.forward(X)
        
        loss = self.loss_fn(pred, Y.long())

        _, classes = torch.max(pred, dim=1)
        correct += torch.sum(classes == Y).item()
        # correct += torch.sum(torch.round(pred).squeeze() == Y)
        total += X.shape[0]

        f1 = f1_score(Y.cpu().detach(), classes.cpu().detach(), average='macro')

        # print(Y.cpu().detach())
        cpupred = pred.cpu().detach()
        
        rocauc = roc_auc_score(Y.cpu().detach(), torch.maximum(cpupred[:, 0], cpupred[:, 1])) 
        accuracy = correct / total

        return loss.item(), accuracy, rocauc, f1
        


    def predict_proba(self, X):
        probs = []
        for x in X:
            probs.append(self.forward(x).item())

        return probs

