import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def test_model(model_path, test_data_path):
    # Load the model
    if not os.path.exists(model_path):
        print(f"{model_path} Model does not exist. Exiting without testing.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the testing data
    test_data = pd.read_csv(test_data_path)

    # Split the data into features and labels
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # Test the model on the data
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    # Calculate the metrics
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    # eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
    far1 = fpr[np.where(tpr == 0)[0][-1]]
    frr1 = 1 - tpr[np.where(fpr == 0)[0][-1]]
    hter = (far1 + frr1) / 2
    far = fpr
    frr = 1 - tpr

    # Detect genuine and imposter scores
    genuine_scores = []
    imposter_scores = []
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 1:
            genuine_scores.append(y_scores[i])
        elif y_test[i] == 0 and y_pred[i] == 1:
            imposter_scores.append(y_scores[i])

    return y_pred, y_scores, eer, far, frr, hter, genuine_scores, imposter_scores

def plot_and_print(model, testfile, title):
    y_pred, y_scores, eer, far, frr, hter, genuine_scores, imposter_scores = test_model(model, testfile)
    print("The scores for " + title + " are:")
    print("EER: ", eer)
    print("HTER: ", hter),
    far1 = far[np.where(far == 0)[0][-1]]
    frr1 = frr[np.where(far == 0)[0][-1]]
    print("FAR: ", far1)
    print("FRR: ", frr1)
    
    # Plot the FAR and FRR

    plt.plot(far, 1 - frr, label='ROC Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve for ' + title)
    plt.legend()
    plt.show()

    # Plot the genuine and imposter scores
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine Scores')
    plt.hist(imposter_scores, bins=50, alpha=0.5, label='Imposter Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Genuine and Imposter Scores for ' + title)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_and_print('small_Webcam_Attack_Model.pkl', 'small_testing.csv', 'a portion of the testing data')
    # plot_and_print('big_Webcam_Attack_Model.pkl', 'big_testing.csv', 'the entire testing data')