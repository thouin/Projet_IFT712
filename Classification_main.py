import sys
import numpy as np
from Classification import Donnees_Wine as data
import Classification_Models as cm

def main():
    if len(sys.argv) < 2:
        print("Usage: python Classification.py model\n")
        print("\t model: Choose a classificattion model\n")
        print(" example: python3 Classification.py Regression_logistique\n")
        return

    model = sys.argv[1]

    # Entrainement du modele de classification
    classification = cm.Regression_Logistique()
    classification.entrainement(data.X_train, data.Y_train)

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = classification.prediction(data.X_train)
    predictions_test = classification.prediction(data.X_test)

    # Calcul des erreurs
    #erreurs_entrainement = np.array([classification.erreur(t_n, p_n) for t_n, p_n in zip(data.Y_train, predictions_train)])
    erreurs_entrainement = classification.erreur(data.Y_train, predictions_train)
    #erreurs_test = np.array([classification.erreur(t_n, p_n) for t_n, p_n in zip(data.Y_test, predictions_test)])
    erreurs_test = classification.erreur(data.Y_test, predictions_test)


    # Affichage des erreurs
    print("Erreur d'entraînement :", "%.4f" % erreurs_entrainement)
    print("Erreur de test :", "%.4f" % erreurs_test)
    print("----------------------------------------------------------")

if __name__ == "__main__":
    main()
