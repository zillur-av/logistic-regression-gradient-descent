# Logistic Regression with Gradient Descent
This repository contains an implementation of logistic regression with gradient descent from scratch in Python. The implementation is done in Jupyter Notebook and provides an example dataset to test the implementation.

# Logistic Regression and Gradient Descent
Logistic regression is a statistical method used to model a binary dependent variable based on one or more predictor variables. In logistic regression, the probability of the dependent variable taking the value 1 is modeled as a function of the predictor variables. This function is typically the sigmoid function:

 $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$ is a linear combination of the predictor variables.

The parameters $\beta_0, \beta_1, \ldots, \beta_n$ are estimated using maximum likelihood estimation. The likelihood function is given by:
$$L(\beta) = \prod_{i=1}^m h_{\beta}(x^{(i)})^{y^{(i)}}(1-h_{\beta}(x^{(i)}))^{1-y^{(i)}}$$
 
where `m` is the number of training examples, $y^{(i)}$ is the true label of the i-th example (either 0 or 1), and $h_{\beta}(x^{(i)})$ is the predicted probability of the i-th example being positive.
The maximum likelihood estimates of the parameters are obtained by minimizing the negative log-likelihood function:

$$J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)}\log(h_{\beta}(x^{(i)})) + (1-y^{(i)})\log(1-h_{\beta}(x^{(i)}))]$$

This is usually done using gradient descent, which is an iterative optimization algorithm that updates the parameter estimates in the direction of steepest descent of the cost function. The update rule for logistic regression with gradient descent is given by:

$$\beta_j := \beta_j - \alpha \frac{\partial J}{\partial \beta_j}$$

where $\alpha$ is the learning rate and $\frac{\partial J}{\partial \beta_j}$ is the partial derivative of the cost function with respect to the $j$-th parameter.
$$\frac{\partial J}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^{m}(h_{\beta}(x^{(i)}) - y^{(i)})x_j^{(i)}$$
# Usage
* Clone this repository: git clone https://github.com/your-username/logistic-regression-with-gradient-descent.git
* Navigate to the project directory: cd logistic-regression-with-gradient-descent
* Open Jupyter Notebook: jupyter notebook
* Run the logistic_regression_gradient_descent.ipynb file
* Follow the instructions provided in the notebook to test the implementation with the provided dataset
# Dependencies
Python 3.x
Numpy
Pandas
Matplotlib
# Credits
The implementation is based on the lecture notes from Andrew Ng's Machine Learning course on Coursera.

# License
This project is licensed under the MIT License. Feel free to use it however you like.
