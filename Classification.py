import pandas as pd
from sklearn.model_selection import train_test_split


class Donnees_Wine():

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


