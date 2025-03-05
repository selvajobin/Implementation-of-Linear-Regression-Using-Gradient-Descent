# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries such as `pandas`, `numpy`, and `sklearn` modules like `StandardScaler`.

2. Load the dataset `50_Startups.csv` using `pandas.read_csv()`.

3. Extract features (`X`) and the target variable (`y`) from the dataset.  
   - `X`: Select all independent variables (features).  
   - `y`: Select the dependent variable (target).

4. Standardize the features `X1` (numerical independent variables in `X`) and the target variable `y` using `StandardScaler` from `sklearn`.  
   - Fit the scaler to `X1` and `y` to compute the mean and standard deviation.  
   - Transform `X1` and `y` to ensure they have zero mean and unit variance.

5. Add an intercept column to `X1`.  
   - Create a column of ones with the same number of rows as `X1`.  
   - Append this column to `X1` to account for the intercept term in the regression.

6. Define a custom function named `linear_regression`.  
   - This function takes the design matrix (`X1`) and the target variable (`y`) as inputs.  
   - Use the formula `theta = (X'X)^(-1) X'y` to compute the learned parameters (`theta`), where `X'` is the transpose of `X`.

7. Call the `linear_regression` function using the standardized `X1` and `y` to obtain the parameters (`theta`).

8. Prepare the new data point for prediction.  
   - Standardize the new data point using the same scaler used for `X1`.  
   - Append the intercept term to the new data point.

9. Make a prediction using the computed `theta`.  
   - Multiply the new data point (including the intercept term) with `theta` to calculate the predicted value.

10. Print the predicted value for the new data point.  
    - Optionally, inverse-transform the standardized prediction to return it to its original scale.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: DEVESH S
RegisterNumber: 212223230041
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv')
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/user-attachments/assets/3a10dc18-4b8f-41e8-9e3b-123e80ea4b0d)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
