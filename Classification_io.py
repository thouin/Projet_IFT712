import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class io():
    @staticmethod
    def getData():
        # Importation des données du dataset 'wine' dans un dataframe nommé: df
        # Renommage des attributs
        df = pd.read_csv("wine.data", header=None)
        df.columns = ["Target","Alcoho","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
        "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
        "OD280/OD315_of_diluted_wines","Proline"]


        # Creation des données d'entrainement et de test
        features = ["Alcoho","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
        "Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
        "OD280/OD315_of_diluted_wines","Proline"]
        Y = df.Target
        X = df.loc[:, features]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
        return X_train, X_test, Y_train, Y_test
    
    @staticmethod
    def plot(train_loss_list, train_accu_list, valid_loss_list, valid_accu_list, filename):
        xdata = np.arange(1, len(loss_train_curve) + 1)
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.ylabel('loss')
        ax1.plot(xdata, loss_train_curve, label='training')
        ax1.plot(xdata, loss_val_curve, label='validation')
        ax1.legend()
        
        ax2.ylabel('accuracy')
        ax2.plot(xdata, loss_train_curve, label='training')
        ax2.plot(xdata, loss_val_curve, label='validation')
        ax2.legend()
        fig.savefig(filename)