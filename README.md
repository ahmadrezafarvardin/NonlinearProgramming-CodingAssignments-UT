# Nonlinear Programming Coding Assignments - UT

This repository contains coding assignments for the Nonlinear Programming course at University of Tehran.

## Repository Structure

```
.
├── CA1/                    # Coding Assignment 1
│   ├── coding_Farvardin_HW1.ipynb
│   ├── Coding_HW1.pdf
│   └── datasets/
│       ├── california_housing.csv
│       ├── data1.csv
│       ├── data2.csv
│       └── diabetes.csv
├── CA2/                    # Coding Assignment 2
│   ├── coding_Farvardin_HW2.ipynb
│   └── NLP_HW2.pdf
└── CA3/                    # Coding Assignment 3
    ├── CA3_NLP.pdf
    ├── HW3.ipynb
    ├── HW3-Q2.ipynb
    ├── gamma.csv
    └── svm.csv
```

## Assignment Descriptions

### CA1: Least Squares and Regularization Methods

This assignment focuses on data fitting using least squares methods and various regularization techniques.

#### Part 1: Polynomial Fitting
- Implementation of least squares fitting on two one-dimensional datasets (`data1.csv` and `data2.csv`)
- Noise removal and function smoothing
- Finding appropriate polynomial degrees for each dataset

#### Part 2: Ridge Regression
- Implementation of Ridge Regression (L2 regularization) on the `diabetes.csv` dataset
- Analysis of the regularization parameter λ effect on the solution
- Comparison with baseline least squares
- Research on Lasso and Elastic Net regression methods

#### Part 3: Weighted Least Squares
- Implementation of weighted least squares with different weight distributions
- Using `california_housing.csv` dataset
- Three weight sampling methods:
  - Uniform distribution (0.5 to 3)
  - Multinomial distribution
  - Dirichlet distribution
- Performance comparison across 100 iterations

### CA2: Optimization Methods

This assignment explores various optimization algorithms for nonlinear functions.

#### Problem 1: Function Optimization
Optimization of f(x) = x₁⁴ + 3x₂² + x₁x₂ using:
- Gradient descent with backtracking line search
- Pure Newton's method
- Newton's method with different line search strategies
- Analysis of parameters α and β effects
- Comparison of convergence rates and iteration counts

#### Problem 2: Polynomial Regression
Implementation of polynomial regression to minimize Mean Squared Error (MSE):
- Gradient and Hessian computation
- Gradient descent implementation
- Newton's method implementation
- Comparison of both methods with different starting points
- Visualization of fitted polynomials

### CA3: Generalized Linear Models and Support Vector Machines

#### Part 1: Gamma Regression
- Derivation of Negative Log-Likelihood for Gamma distribution
- Implementation using three optimization methods:
  - Convex optimization with cvxpy
  - Manual gradient descent
  - Manual Newton-Raphson method
- Comparison with statsmodels implementation
- 10-fold cross-validation evaluation

#### Part 2: Support Vector Machines
Implementation and comparison of three SVM variants:
- Hard-Margin SVM
- Soft-Margin SVM  
- Kernel SVM

Each variant is implemented both manually using cvxpy and using scikit-learn, with performance evaluation using:
- Precision
- Recall
- F1-Score
- Accuracy

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib/Seaborn
- cvxpy
- scikit-learn
- statsmodels
- Jupyter Notebooks

## Course Information
- **Course**: Nonlinear Programming
- **University**: University of Tehran
- **Student**: Ahmadreza Farvardin

## Requirements
To run the notebooks, you'll need to install the required packages:
```bash
pip install numpy pandas matplotlib seaborn cvxpy scikit-learn statsmodels jupyter
```

## Usage
1. Clone the repository
2. Navigate to the desired assignment folder (CA1, CA2, or CA3)
3. Open the Jupyter notebooks to view the implementations
4. Ensure all required datasets are in their respective directories

## Note
The assignment descriptions and problem statements are originally in Persian. The implementations and code comments are in English.
