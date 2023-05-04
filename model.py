from torch.nn import *
from torch import optim
import torch
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt

from data import get_default_device, to_device

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
                Linear(15, 1),
                Sigmoid(),
                )


        self.loss_fn = None
        self.optim_fn = None
        
        if self.optim_fn == None:
            self.optim_fn = optim.Adam(self.parameters(), lr=self.lr)
        if self.loss_fn == None:
            self.loss_fn = BCELoss() 

    def forward(self, X):
        return self.layer(X)


    def fit(self, train_dl, val_dl):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        ########################################################################

        train_accuracies = []
        train_losses = []
        validation_accuracies = []
        validation_losses = []


        for epoch in range(self.epochs):

            val_accuracy = None
            train_accuracy = None
            total = 0
            correct = 0

            tr_losses = []
            val_losses = []
            train_loss = 0
            val_loss = 0

            for i, (x, y) in enumerate(train_dl):
                pred = self(x)

                self.optim_fn.zero_grad()

                loss = self.loss_fn(pred.reshape(-1), y.float())

                loss.backward()

                self.optim_fn.step()
                
                tr_losses.append(loss.item())
                # _, classes = torch.max(pred, dim=1)
                correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += pred.size(0)

                if epoch == self.epochs-1 and i == 0:
                #     prec, rec, thresh = precision_recall_curve(y.cpu().detach(), pred.squeeze().cpu().detach())
                #     plt.title("Precision Recall Curve")
                #     plt.plot(rec, prec)
                #     plt.xlabel("Recall")
                #     plt.ylabel("Precision")
                #     plt.show()
                    print(f"AUC-ROC Score: {roc_auc_score(y.cpu().detach(), pred.squeeze().cpu().detach()) :f}")


            train_accuracy = correct / total
            train_loss = np.mean(tr_losses)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            total = 0
            correct = 0

            for (x, y) in val_dl:
                pred = self.forward(x)

                loss = self.loss_fn(pred.reshape(-1), y.float())
                val_losses.append(loss.item())

                self.optim_fn.step()

                # _, classes = torch.max(pred, dim=1)
                # correct += torch.sum(classes == y).item()
                correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += x.size(0)

            val_loss = np.mean(val_losses)
            val_accuracy = correct / total
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)


            # if (epoch + 1) % 5 == 0:  
            #     # Print progress
            #     if val_accuracy is not None:
            #         print("Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_accuracy: {:.4f}, val_accuracy: {:.4f}"
            #                   .format(epoch+1, self.epochs, train_loss, val_loss, train_accuracy, val_accuracy))
            #     else:
            #         print("Epoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.4f}"
            #                   .format(epoch+1, self.epochs, train_loss, train_accuracy))

        return train_losses, train_accuracies, validation_losses, validation_accuracies
            

    def getMetrics(self, X, Y):

        total = 0
        correct = 0
        losses = []

        pred = self.forward(X)
        
        loss = self.loss_fn(pred.squeeze(), Y)

        # _, classes = torch.max(pred, dim=1)
        # correct += torch.sum(classes == y).item()
        correct += torch.sum(torch.round(pred).squeeze() == Y)
        total += X.shape[0]
        
        rocauc = roc_auc_score(Y.cpu().detach(), pred.squeeze().cpu().detach()) 
        accuracy = correct / total

        return loss.item(), accuracy.cpu().detach(), rocauc
        


    def predict_proba(self, X):
        probs = []
        for x in X:
            probs.append(self.forward(x).item())

        return probs

