import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification

def generate_multilabel_dataset(n_rows, n_features, target_column_name, n_classes_per_label):
    X, y = make_multilabel_classification(n_samples=n_rows, n_features=n_features, n_classes=n_classes_per_label, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df[target_column_name] = [','.join(map(str, classes)) for classes in y]
    return df

def split_multilabel_dataset(n_rows, n_features, target_column_name, n_classes_per_label, test_size, n_splits):
    train_test_splits = []
    for i in range(n_splits):
        df = generate_multilabel_dataset(n_rows, n_features, target_column_name, n_classes_per_label)
        train, test = train_test_split(df, test_size=test_size)
        train.to_csv(f'data/c{i+1}/train_microbiome_raw.csv', index=False, sep=',')  # Export train dataset to CSV with semicolon separator
        test.to_csv(f'data/c{i+1}/test_microbiome_raw.csv', index=False, sep=',')  # Export test dataset to CSV with semicolon separator
        train_test_splits.append((train, test))
    return train_test_splits

# Example usage
n_rows = 1000  # Number of rows
n_features = 1000  # Number of features
target_column_name = "health_status"  # Target column name
n_classes_per_label = 3  # Number of classes per label
test_size = 0.2  # 20% test data
n_splits = 10  # Number of splits

train_test_splits = split_multilabel_dataset(n_rows, n_features, target_column_name, n_classes_per_label, test_size, n_splits)
for i, (train, test) in enumerate(train_test_splits):
    print(f"Split {i+1}: Train size - {len(train)}, Test size - {len(test)}")
