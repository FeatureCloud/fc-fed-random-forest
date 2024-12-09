# Each subfolder has test.csv and pred.csv, calc accuracy from them and print it
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
scriptfile_folder = os.path.dirname(os.path.abspath(__file__))

def check_accuracy(test_path, pred_path):
    test = pd.read_csv(test_path)
    pred = pd.read_csv(pred_path)
    acc = accuracy_score(test['y_true'], pred['pred'])
    return acc

def main():
    for folder in os.listdir(scriptfile_folder):
        if not os.path.isdir(os.path.join(scriptfile_folder, folder)):
            continue
        test_path = os.path.join(scriptfile_folder, folder, 'test.csv')
        pred_path = os.path.join(scriptfile_folder, folder, 'pred.csv')
        acc = check_accuracy(test_path, pred_path)
        print(f'{folder}: {acc}')

if __name__ == '__main__':
    main()