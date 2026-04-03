import tensorflow as tf
import numpy as np
import idx2numpy
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def load_test_data():
    X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    
    # Normalize and reshape
    X_test = X_test / 255.0
    X_test = X_test.reshape((-1, 28, 28, 1))
    
    return X_test, y_test

def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()
    
    print("Loading trained model...")
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    
    print("Making predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert to DataFrame for a clean table
    df_report = pd.DataFrame(report).transpose()
    
    # Format the table
    print("\n" + "="*50)
    print("ACCURACY REPORT PER DIGIT (0-9)")
    print("="*50)
    
    # Select only digits 0-9 for the detailed table
    digit_rows = [str(i) for i in range(10)]
    detailed_report = df_report.loc[digit_rows].copy()
    detailed_report.index.name = 'Digit'
    
    # Rename columns for clarity (optional)
    detailed_report.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    
    print(detailed_report.to_string())
    
    print("\n" + "="*50)
    print(f"OVERALL ACCURACY: {report['accuracy']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
