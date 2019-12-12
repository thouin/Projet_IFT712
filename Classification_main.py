import sys
import numpy as np
from Classification_io import io
import Classification_Models as cm

def main():
    if len(sys.argv) < 2:
        print("Usage: python Classification.py model\n")
        print("\t model: Choose a classification model\n")
        print("\t models to choose: 'Regression_logistique' or 'SVM_Sigmoide_Kernel' \n")
        print(" example: python3 Classification.py Regression_logistique\n")
        return

    model = sys.argv[1]

    # Entrainement du modele de classification
    if model =='Regression_logistique' :
        classification = cm.Regression_Logistique()
    if model =='SVM_Sigmoide_Kernel' :
        classification = cm.SVM_Sigmoide_Kernel()

    X_train, X_test, Y_train, Y_test = io.getData()
    classification.entrainement(X_train, Y_train)

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = classification.prediction(X_train)
    predictions_test = classification.prediction(X_test)

    # Calcul des erreurs
    #erreurs_entrainement = np.array([classification.erreur(t_n, p_n) for t_n, p_n in zip(data.Y_train, predictions_train)])
    erreurs_entrainement = classification.erreur(Y_train, predictions_train)
    #erreurs_test = np.array([classification.erreur(t_n, p_n) for t_n, p_n in zip(data.Y_test, predictions_test)])
    erreurs_test = classification.erreur(Y_test, predictions_test)


    # Affichage des erreurs
    print("Erreur d'entraînement :", "%.4f" % erreurs_entrainement)
    print("Erreur de test :", "%.4f" % erreurs_test)
    print("----------------------------------------------------------")

if __name__ == "__main__":
    main()
