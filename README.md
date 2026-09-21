# Linear Regression from Scratch

Implementation of simple linear regression using **gradient descent**, written
from scratch in Python and compared against scikit-learn's `LinearRegression`.

## Problem

Predict a person's salary from their years of experience
(`YearsExperience` → `Salary`).

## What's inside

- `regrsion_lineal.py`: manual implementation
  - `cost_function`: Mean Squared Error / 2
  - `gradient_function`: partial derivatives ∂J/∂w and ∂J/∂b
  - `gradient_descent`: iterative parameter update `w := w - α·∂J/∂w`
- `regresion_Lineal_libreria.py`: same model with scikit-learn
- `dataset_train.csv`: training data

## Model

ŷ = w·x + b

Cost: J(w, b) = (1 / 2m) · Σ (ŷᵢ − yᵢ)²

## How to run

    pip install numpy pandas matplotlib scikit-learn
    python regrsion_lineal.py
    python regresion_Lineal_libreria.py

## Results

| Method                        | w   | b   |
| ----------------------------- | --- | --- |
| Gradient descent (manual)     | ... | ... |
| scikit-learn LinearRegression | ... | ... |

_Add the fitted-line plot and the cost-vs-iterations curve here._

## What I learned

- How gradient descent minimizes a cost function step by step
- Effect of the learning rate and number of iterations on convergence
- Why the manual result should match the closed-form solution of scikit-learn

## Next steps

- Train/test split and metrics (MSE, R²)
- Feature scaling and vectorized multiple regression
- Regularization (Ridge / Lasso)
