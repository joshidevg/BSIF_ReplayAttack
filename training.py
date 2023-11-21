import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier
import time

def train_model(training_file_path, model_file_path):
    # Load the training data from the csv file
    training_data = pd.read_csv(training_file_path)
    
    # Split the data into features and labels
    X = training_data.iloc[:, :-1]
    y = training_data.iloc[:, -1]
    
    # Train an Extra Trees Classifier model on the data
    model = ExtraTreesClassifier()
    startt = time.time()
    model.fit(X, y)
    endt = time.time()

    print("Training time: ", endt - startt, " seconds for ", training_file_path, " file.")

    # Save the trained model as a pickle file
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    # train_model('small_training.csv', 'small_Webcam_Attack_Model.pkl')
    # train_model('big_training.csv', 'big_Webcam_Attack_Model.pkl')
    train_model('small_bsif_13_training.csv', 'small_BSIF_13_Webcam_Attack_Model.pkl')
    train_model('small_bsif_17_training.csv', 'small_BSIF_17_Webcam_Attack_Model.pkl')
    train_model('small_bsif_21_training.csv', 'small_BSIF_21_Webcam_Attack_Model.pkl')
    train_model('big_bsif_13_training.csv', 'big_BSIF_13_Webcam_Attack_Model.pkl')
    train_model('big_bsif_17_training.csv', 'big_BSIF_17_Webcam_Attack_Model.pkl')
    train_model('big_bsif_21_training.csv', 'big_BSIF_21_Webcam_Attack_Model.pkl')