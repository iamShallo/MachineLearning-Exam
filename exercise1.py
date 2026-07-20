import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def main():
    # Load dataset
    df = pd.read_csv('regression1.csv', header=None)
    X = df.iloc[:, 0].values.reshape(-1, 1)
    y = df.iloc[:, 1].values
    
    # Using k-NN to handle non-linear discontinuities
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    
    # Create prediction grid
    x_min, x_max = X.min(), X.max()
    X_interval = np.arange(x_min, x_max, 0.25).reshape(-1, 1)
    
    # Run predictions
    y_predicted = model.predict(X_interval)
    
    # Save output
    results_df = pd.DataFrame({'x': X_interval.flatten(), 'y': y_predicted})
    results_df.to_csv('result1.csv', index=False)

if __name__ == '__main__':
    main()