from torch.nn import *
from torch import optim


class Model(nn.Module):
    def __init__(self, lr=0.001, n_epochs=5):
        super(Model, self).__init__()

        self.lr = lr
        self.epochs = n_epochs


        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        ########################################################################

        self.l1 = Sequential(
                    Linear(32, 64),
                    ReLU()
                )

        self.l2 = Sequential(
                    Linear(64, 1),
                    LeakyReLU()
                )
        
        pass


    def forward(self, X):

        x = self.l1(X)
        return self.l2(x)


    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        ########################################################################

        self.optim = optim.Adam(self.parameters, lr=self.lr)
        self.loss = MSELoss()
       
        for epoch in range(self.epochs):
            



    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x
        pass
        ########################################################################
        #return preds
