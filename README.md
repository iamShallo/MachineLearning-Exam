# Machine Learning Homework: Regression and Classification

This repository contains the machine learning solutions for the **Fundamentals of Machine Learning** course, developed by **Francesco Caldarelli**. The project involves building and implementing regression and classification models to analyze specific datasets.

## Exercises Overview

| Exercise | Task Type | Source Dataset | Script | Output Result |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Regression | `regression1.csv` | `exercise1.py` | `result1.csv` |
| 2 | Classification | `classification1.csv` | `exercise2.py` | `result2.csv` |
| 3 | Regression | `regression2.csv` | `exercise3.py` | `result3.csv` |
| 4 | Classification | `classification2.csv` | `exercise4.py` | `result4.csv` |

## Exercise Descriptions

### Exercise 1: Non-linear Regression
The objective was to model a dataset characterized by a sharp geometric discontinuity. I implemented a **k-Nearest Neighbour (k-NN) Regressor** (with $k=3$) to capture the rapid change in the function's values, which standard parametric models failed to represent accurately.

### Exercise 2: Multiclass Classification
This task involved classifying points based on a dataset with two features. I used a **Multinomial Logistic Regression** model to define the decision boundaries. The model effectively separates the classes and generalizes the underlying distribution of the provided data.

### Exercise 3: Bivariate Surface Regression
The goal was to model an unknown bivariate function. I utilized a **k-NN Regressor** to map the complex, step-like surface of the data. This approach allows the model to handle local spatial dependencies and abrupt transitions in the bivariate function effectively.

### Exercise 4: Complex Classification
This dataset presented significant noise and class overlapping. To address this, I implemented a **k-NN Classifier** with a larger neighborhood ($k=21$). This choice smooths out the local noise and creates more robust decision boundaries, improving the overall classification accuracy.

## Requirements
- Python 3.x
- Dependencies: `pip install scikit-learn pandas numpy`

## How to run
Ensure the dataset files are in the same directory as the scripts and run:
- `python exercise1.py`
- `python exercise2.py`
- `python exercise3.py`
- `python exercise4.py`

## Author
**Francesco Caldarelli** - University of Camerino
