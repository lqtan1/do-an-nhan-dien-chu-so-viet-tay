import tensorflow as tf
import numpy as np
import idx2numpy
import pandas as pd
from sklearn.metrics import classification_report

def load_test_data():
    X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte') / 255.0
    X_test = X_test.reshape((-1, 28, 28, 1))
    y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    return X_test, y_test

def main():
    X_test, y_test = load_test_data()
    model = tf.keras.models.load_model('mnist_cnn_model.h5')
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()[:10]
    df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
    
    with open('accuracy_table.md', 'w', encoding='utf-8') as f:
        f.write('# Accuracy Report\\n\\n')
        f.write(df.to_markdown())
        f.write(f'\\n\\n**Overall Accuracy**: {report["accuracy"]:.4f}\\n')

if __name__ == "__main__":
    main()
