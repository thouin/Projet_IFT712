import pandas as pd
from sklearn.model_selection import train_test_split


#Importation des données du dataset 'wine' dans un dataframe nommé: df
#Renommage des attributs
df = pd.read_csv("wine.data", header=None)
df.columns = ["Target","Alcoho","Malic_acid","Ash","Alcalinity_of_ash","Magnesium","Total_phenols",
 	"Flavanoids","Nonflavanoid_phenols","Proanthocyanins","Color_intensity","Hue",
 	"OD280/OD315_of_diluted_wines","Proline"]

#Creation des données d'entrainement et de test
Y = df['Target']
X = df
X.drop(['Target'], axis='columns', inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
