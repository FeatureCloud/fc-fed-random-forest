import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_synthetic_dataset(n, rows, features, classes= None, problem_type='classification'):
    for i in range(n):
        data = pd.DataFrame(data= np.random.rand(rows, features),
                            columns= [f'Feature_{j+1}' for j in range(features)])
        if problem_type == 'classification':
            data['health_status'] = np.random.choice(classes, size=rows)
        elif problem_type == 'regression':
            # Adjust range as needed for your regression targets
            data['health_status'] = np.random.rand(rows) * 100
        # data.to_csv(f'data_{i+1}.csv', index=False)

        # Split the dataset into training and testing sets with a test size of 0.25
        print(f'Splitting dataset {i+1}')
        split_train_test(data, 0.25, i+1)
def split_train_test(data, test_size, i):
    X = data.drop(columns=['health_status'])
    y = data['health_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(f'data/c{i}/train_microbiome_raw.csv', index=False)
    test.to_csv(f'data/c{i}/test_microbiome_raw.csv', index=False)

# Create synthetic dataset for classification problem
# Each synthetic dataset will be created with:
# 3 files, each having 7500 rows, 1000 features, and 3 classes for the health_status variable
create_synthetic_dataset(n=10, rows=7500, features= 1000, classes= [0, 1])

# Create synthetic dataset for regression problem
# create_synthetic_dataset(n=3, rows=7500, features= 1000, problem_type='regression')
