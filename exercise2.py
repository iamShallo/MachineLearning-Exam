import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
    # Load features and labels
    df = pd.read_csv('classification1.csv', header=None)
    X_train = df.iloc[:, :2].values
    y_train = df.iloc[:, 2].values
    
    # Logistic regression handles multiclass automatically in newer sklearn versions
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Define grid range [-45, 45]
    x1_range = np.arange(-45.0, 45.25, 0.25)
    x2_range = np.arange(-45.0, 45.25, 0.25)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    grid_points = np.c_[X1.ravel(), X2.ravel()]
    
    # Predict and format output
    predictions = model.predict(grid_points)
    results_df = pd.DataFrame({
        0: grid_points[:, 0],
        1: grid_points[:, 1],
        2: predictions
    })
    
    results_df.to_csv('result2.csv', index=False, header=False)

if __name__ == '__main__':
    main()