# preethamchippa-EE658-758-Machine-Learning-in-Engineering-Assignment-2--Logistic--Regression

### Logistic Regression with Non-linear Decision Boundary
This project explores various approaches to creating predictive models using logistic regression for a dataset with two classes that are not easily separable by a linear decision boundary. The project is divided into four parts.

**Part A: Logistic Regression with Scikit-learn**
In this part, we load the dataset data.csv and split it into training and testing subsets. Then, we develop a logistic regression model based on the training data using scikit-learn, utilizing only Feature1 and Feature2. After that, we plot the data points and the decision boundary of the model. Finally, we evaluate the model on the testing data and report its accuracy.

**Part B: Logistic Regression Without Scikit-learn**
Here, we implement logistic regression from scratch. We use the same training and testing data splits as in Part A and develop a logistic regression model based on the training data using only Feature1 and Feature2. We display the cost (loss) as a function of iterations during the training process. Then, we plot the data points and the decision boundary of our model. Finally, we evaluate the model on the testing data and report its accuracy.

**Part C: Logistic Regression with Feature Engineering (Scikit-learn)**
In this part, we enhance the dataset by adding new features that are at a higher degree of one of the original features (e.g., Feature1^2, Feature1 * Feature2). We use scikit-learn to develop a logistic regression model based on the enhanced training data. We plot the data points and the decision boundary of the model and evaluate its performance on the testing data, reporting its accuracy.

**Part D: Logistic Regression with Feature Engineering (Without Scikit-learn)**
Here, we use the enhanced dataset from Part C and implement logistic regression from scratch to develop a model based on the enhanced training data. We display the cost (loss) as a function of iterations during the training process, plot the data points and the decision boundary of our model, and evaluate its performance on the testing data, reporting its accuracy.


# EE658-758-Machine-Learning-in-Engineering-Assignment-1-Linear-Regression

**Overview**
This project focuses on analyzing expenditures on insured customers for an insurance company using machine learning techniques. The primary objective is to forecast the financial outlay incurred by the insurance company on these customers based on various demographic and lifestyle factors. The dataset comprises profiles of insured clients, including information such as age, gender, BMI (Body Mass Index), number of children, smoking status, region, and medical expenses.

Tasks Overview

**Data Preprocessing:**

Load the dataset and handle any missing values by removing corresponding rows.
Encode categorical variables such as gender, smoking status, and region into numerical values suitable for regression analysis.
Normalize the features using Min-Max scaling to ensure uniformity in the range of values.

**Splitting the Data:**

Divide the data into "features" and "target" subsets.
Split the dataset into training and testing subsets to evaluate model performance.

**Gradient Descent Implementation:**

Implement the gradient descent algorithm from scratch to find the regression line.
Experiment with both constant and decaying learning rates, recording loss values for each iteration.
Show coefficients and intercept of the model obtained through gradient descent.

**Model Evaluation:**

Predict medical expenses for the testing dataset using the trained model.
Compute Mean Absolute Error (MAE) and Mean Squared Error (MSE) to assess prediction accuracy.
Visualize the error distribution using a histogram.

**Learning Rate Analysis:**

Analyze the effect of varying learning rates on the convergence of the gradient descent algorithm.

**Scikit-learn Implementation:**

Utilize the linear_model.LinearRegression class from scikit-learn to repeat the regression.
Compute MAE and MSE for comparison with the custom implementation.

**Normal Equation Implementation:**

Implement the normal equation method to find the regression line directly.
Compare the MAE and MSE with previous methods to assess performance.

**Comparison:**

Compare the three solutions in terms of MAE, MSE, and computational efficiency to identify the most effective approach.
