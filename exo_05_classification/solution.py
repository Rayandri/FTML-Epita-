import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

def main():
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    model = SVC(
        kernel='poly',
        C=0.004692951967253321,
        gamma=0.1598562868324287,
        degree=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {accuracy:.4f}")
    
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, accuracy

if __name__ == "__main__":
    model, accuracy = main() 
