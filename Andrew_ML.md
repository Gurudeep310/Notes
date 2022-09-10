---
Title: 'Machine Learning'
Author: Gurudeep
Date: '13th March 2022'
---  
  
<center><h1>Machine Learning</h1></center>
<h3 style = "text-align:right; font-family:Roboto">-By Gurudeep</h3>
<p>References from Andrew NG Lectures</p>
  

TABLE OF CONTENTS

[TOC] 

```
{ignore = true}
```

## What is machine Learning ?
Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

Example: playing checkers.
E = the experience of playing many games of checkers
T = the task of playing checkers.
P = the probability that the program will win the next game.

In general, any machine learning problem can be assigned to one of two broad classifications:
Supervised learning and Unsupervised learning.

**Supervised Learning**
In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

Eg:
1. Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
2. Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign. 

**Unsupervised Learning**
Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback based on the prediction results.

Eg:
1. Clustering: Take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles, and so on.
2. Non-clustering: The "Cocktail Party Algorithm", allows you to find structure in a chaotic environment.

## Model Representation
To establish notation for future use, we’ll use $x^{(i)}$to denote the “input” variables (living area in this example), also called input features, and $y^{(i)}$ to denote the “output” or target variable that we are trying to predict (price). A pair $(x^{(i)} , y^{(i)} )$ is called a training example

When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem.

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test1.jpg" alt = "drawing" width="700"/></center><br>

**Cost Function**
$$J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_i) - y_i)^2$$

>1. Hypothesis
$h_{\theta}(x) = \theta_0 + \theta_1x$
>2. Parameters 
$\theta_1,\theta_2$
>3. Cost Function
$J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x_i) - y_i)^2$
$m$ is total number of samples
>4. Goal
Min $J$ by Applying Gradient Descent Algorithm $\rightarrow$ Batch Gradient Descent Algorithm

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test2.jpg" alt = "drawing" width="700"/></center><br>

## Parameter Learning

### Gradient Descent
Gradient Descent to min J
Gradient Descent Algorithm
$$\theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$$ for j = 0 and j = 1
$\alpha = $ Learning Rate

**Correct Simulatenous Update**
$temp0 := \theta_0 - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$
$temp1 := \theta_1 - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$
$\theta_0 := temp0$
$\theta_1 := temp1$

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test2.jpg" alt = "drawing" width="700"/></center><br>

Gradient Descent can converge to a local minimum, even with the learning rate $\alpha$ fixed. As we approach local minimum, gradient descent will automatically take smaller steps. So no need to decrease $\alpha$ over time.

### Batch Gradient Descent For Linear Regression

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function J. This method looks at every example in the entire training set on every step, and is called batch gradient descent. 

$\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}) = \frac{\partial}{\partial \theta_0}J(\theta_0,\theta_1)$
$\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x^{(i)} = \frac{\partial}{\partial \theta_1}J(\theta_0,\theta_1)$

repeat until convergence{
    $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})$
    $\theta_1 = \theta_1 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x^{(i)}$
}

## Matrices

### Matrices
```matlab
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)

% Output
A =
    1    2    3
    4    5    6
    7    8    9
   10   11   12

v =
   1
   2
   3

m =  4
n =  3
dim_A =
   4   3

dim_v =
   3   1

A_23 =  6
```

```matlab
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s

%Output
A =
   1   2   4
   5   3   2

B =
   1   3   4
   1   1   1

s =  2
add_AB =
   2   5   8
   6   4   3

sub_AB =
   0  -1   0
   4   2   1

mult_As =
    2    4    8
   10    6    4

div_As =
   0.50000   1.00000   2.00000
   2.50000   1.50000   1.00000

add_As =
   3   4   6
   7   5   4
```
<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test4.jpg" alt = "drawing" width="700"/></center><br>

```matlab
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v

% Output
A =
   1   2   3
   4   5   6
   7   8   9

v =
   1
   1
   1

Av =
    6
   15
   24
```
### Matrix Matrix Multiplication
<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test5.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test6.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test7.jpg" alt = "drawing" width="700"/></center><br>

```matlab
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B

% Make sure you understand why we got that result

% Output
A =
   1   2
   3   4
   5   6

B =
   1
   2

mult_AB =
    5
   11
   17
```

### Matrix Multiplication Properties

1. $A * B != B*A$ Not commuatative

2. $A*(B*C) = (A*B)*C$ Associative

3. $A*I = I*A = A$ where $I =$ Identity Matrix

```matlab
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA

% Output
A =
   1   2
   4   5

B =
   1   1
   0   2

I =
Diagonal Matrix
   1   0
   0   1

IA =
   1   2
   4   5

AI =
   1   2
   4   5

AB =
    1    5
    4   14

BA =
    5    7
    8   10
```

### Matrix Inverse and Transpose
If $A$ is an $m*m$ matrix and if it has inverse then:
$$AA^{-1} = A^{-1}A = I$$
```matlab
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A

% Output
A =
   1   2   0
   0   5   6
   7   0   9

A_trans =
   1   0   7
   2   5   0
   0   6   9

A_inv =
   0.348837  -0.139535   0.093023
   0.325581   0.069767  -0.046512
  -0.271318   0.108527   0.038760

A_invA =
   1.00000  -0.00000   0.00000
   0.00000   1.00000  -0.00000
  -0.00000   0.00000   1.00000
```
## Multi Variate Linear Regression

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test8.jpg" alt = "drawing" width="700"/></center><br>

Hypothesis: $h_\theta = \theta_0 + \theta_1*x_1 + \theta_2*x_2 + \theta_3*x_3 + \theta_4*x_4$

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test9.jpg" alt = "drawing" width="700"/></center><br>

Note: $\theta^T$ is a 1 by (n+1) matrix

## Linear Regression Terms
>1. Hypothesis $h_{\theta} = \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$
>2. Parameters: $\theta_0,\theta_1,...,\theta_n = \theta \rightarrow$ n + 1 dimensional vector.
>3. Cost Function: $J(\theta_0,\theta_1,..,\theta_n) = J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$
>4. Gradient Descent:
Repeat{
    $\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0,...,\theta_n)$
    (simultaneously update for every j = 0,...,n)
}

Previously for n=1
repeat until convergence{
    $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})$
    $\theta_1 = \theta_1 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x^{(i)}$
}

Now for n >= 1
repeat until convergence{
    $\theta_0 = \theta_0 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})$
    $\theta_1 = \theta_1 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_1^{(i)}$
    $\theta_2 = \theta_2 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_2^{(i)}$
    ....
    ....
    ....
}

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test10.jpg" alt = "drawing" width="700"/></center><br>

### Feature Scaling for Gradient Descent
- Idea: Make sure features are on a similar scale.
- Get every feature into approximately a -1 <= $x_i$ <= 1 range.

**Mean Normalisation**
Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean.(Do not apply to $x_0 = 1$)
$X_1 \leftarrow \frac{X_1 - \mu_1}{S_1}$
$\mu_1 \rightarrow$ avg value of $X_1$ in training set.
$S_1 \rightarrow$ range(max-min) or std dev

<u>*This helps gradient descent converge faster*</u>

### Learning rate $\alpha$ for Gradient Descent

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test11.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test12.jpg" alt = "drawing" width="700"/></center><br>


## Normal Equation
 Gives a method to solve $\theta$ analytically instead of iteratively getting $\theta$ using Gradient Descent Method.
 $$\theta = (X^TX)^{-1}X^Ty$$

 <center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test13.jpg" alt = "drawing" width="700"/></center><br>

 <center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test14.jpg" alt = "drawing" width="700"/></center><br>

With the normal equation, computing the inversion has complexity $O(n^3)$. So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

What if $(X^TX)^{-1}$ is non Invertible?
- Then it means there are redudant features, where two features are very closely related(linearly dependant)
    
    - Eg: $x_1 = $ size of in $feet^2$
        $x_2 = $ size in $m^2$
- Too many features(eg: m<=n)
    - Delete some features or use regularisation.

## Classification
### Linear Regression as Classification

- Email: Spam/Not spam

- Online Transactions: Fradulent (Yes/No)?

- Tumor: Malingnant/Benign?

$y \in (0,1)$ $0:$ Negative Class ; $1:$ Positive Class

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test15.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test16.jpg" alt = "drawing" width="700"/></center><br>

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1.

$\therefore$ Linear Regression for classification problem isn't a good idea. Even though we need $y = 1$ or $y = 0; h_{\theta}$ can be $>1$ or $<0$

$\therefore$ We use Logistic Regression: $0\leq h_{\theta}(x)\leq1$.

### Logistic Regression

**Hypothesis Representation :**
We use Logistic Regression: $0\leq h_{\theta}(x)\leq1$.
$h_{\theta}(x) = g(\theta^Tx)$ where $g(z) = \frac{1}{1+e^{-z}} \rightarrow$ Sigmoid Function

Hypothesis: $h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}}$
Interpretation of Hypothesis: $h_{\theta}(x)=$ estimated probability that y = 1 on input x

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test17.jpg" alt = "drawing" width="700"/></center><br>

**Decision Boundary :**

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test18.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test19.jpg" alt = "drawing" width="700"/></center><br>

**Non Linear Decision Boundaries :**

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test20.jpg" alt = "drawing" width="700"/></center><br>

**Cost Function:**
How to choose parameters $\theta ?$
Linear Regression: $J(\theta) = \frac{1}{m}\sum_{i=1}^m \frac{1}{2}(h_{\theta}(x^{(i)})-y^{(i)})^2$

Let $\frac{1}{2}(h_{\theta}(x^{(i)})-y^{(i)})^2 = Cost(h_{\theta}(x),y)$
If we use same Cost/Loss function for Logistic Regression we get a non convex function which may not provide local minimum.

Logistic Regression: 
$ Cost(h_{\theta}(x),y) = -log(h_{\theta}(x))$ if $y=1$
$ Cost(h_{\theta}(x),y) = -log(1-h_{\theta}(x))$ if $y=0$

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test21.jpg" alt = "drawing" width="700"/></center><br>

**Simplified Cost Function**
$Cost(h_{\theta}(x),y) = -y*log(h_{\theta}(x))-(1-y)*log(1-h_{\theta}(x))$
$\therefore J(\theta) = -\frac{1}{m}[\sum_{i=1}^m -y^{(i)}*log(h_{\theta}(x^{(i)}))-(1-y)*log(1-h_{\theta}(x^{(i)}))]$

### Logistic Regression Terms

>- Hypothesis: $h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}}$
>- Parameters: $\theta$
>- Cost Function:$J(\theta) = -\frac{1}{m}[\sum_{i=1}^m -y^{(i)}*log(h_{\theta}(x^{(i)}))-(1-y)*log(1-h_{\theta}(x^{(i)}))]$
>- Gradient Descent: $\theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1)$
repeat until convergence{
    $\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_j^{(i)}$
    ( j = 0,1,2,3,4,...,n)
} $\rightarrow$ Algorithm looks identical to linear regression
> A vectorized implementation is:
> $\theta := \theta - \frac{\alpha}{m}X^T(g(X\theta)-\vec y)$

**Optimisation for Logistic Regression:**
<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test22.jpg" alt = "drawing" width="700"/></center><br>

#### MultiClass Classification - One Vs All
Email foldering/tagging: Work,Friends,Family,Hobby
Medical diagrams: Not ill, Cold, Flu
Weather: Sunny, Cloudy, Rain, Snow

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test23.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test24.jpg" alt = "drawing" width="700"/></center><br>

We train a logistic regression classifier $h_{\theta}^{(i)}(x)$ for each class $i$ to predict the probability that $y = i$
On new input $x$, to make a prediction, pick the class $i$ that maximizes $max_i h_{\theta}^{(i)}(x)$

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test25.jpg" alt = "drawing" width="700"/></center><br>

## Problem of Overfitting
### Overfitting
<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test24.jpg" alt = "drawing" width="700"/></center><br>

<div style = "display:flex; flex-direction: row; justify-content: space-between;">
    <img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test26.jpg" alt = "drawing" style="width:50%"/>
    <img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test27.jpg" alt = "drawing" style="width:50%"/>
</div>

<br>

**Addressing Overfitting**

1. Reduce Number of features

    - Manually select which features to keep.
    
    - Model selection algorithm

2. Regularisation
    
    - Keep all the features, but reduce magnitude/values of 
    parameters $\theta_j$
    
    - Works well when we have a lot of features, each of which contributes a bit to predicting $y$

### Cost Function with Regularisation

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test28.jpg" alt = "drawing" width="700"/></center><br>

**Regularisation:**
 Small values for parameters $\theta_0,....,\theta_n$

Cost Function: $J(\theta) = \frac{1}{2m}[\sum_{i=1}^m (h_{\theta}(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j= 1}^n \theta_j^2]$
If $\lambda$ is very very large it may lead to underfitting.

### Regularised Linear Regression Terms

>1. Hypothesis $h_{\theta} = \theta^Tx = \theta_0x_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$
>2. Parameters: $\theta_0,\theta_1,...,\theta_n = \theta \rightarrow$ n + 1 dimensional vector.
>3. Cost Function: $J(\theta_0,\theta_1,..,\theta_n) = J(\theta) = \frac{1}{2m}[\sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2+ \lambda \sum_{j= 1}^n \theta_j^2]$
>4. Gradient Descent:
Repeat  {
    $\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_0^{(i)}$
    $\theta_j := \theta_j - \alpha[\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_j^{(i)} + \frac{\lambda}{m}\theta_j]$
    ( j = 1,2,3,4,...,n) # NOTE
    (simultaneously update)
}

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test29.jpg" alt = "drawing" width="700"/></center><br>

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test30.jpg" alt = "drawing" width="700"/></center><br>

### Regularised Logistic Regression Terms

>- Hypothesis: $h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}}$
>- Parameters: $\theta$
>- Cost Function:$J(\theta) = -\frac{1}{m}[\sum_{i=1}^m -y^{(i)}*log(h_{\theta}(x^{(i)}))-(1-y)*log(1-h_{\theta}(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j= 1}^n \theta_j^2]$
>- Gradient Descent:
repeat until convergence{
    $\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_0^{(i)}$
    $\theta_j := \theta_j - \alpha[\frac{1}{m}\sum_{i = 1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)}).x_j^{(i)} +\frac{\lambda}{m}\theta_j]$
    ( j = 1,2,3,4,...,n)
} $\rightarrow$ Algorithm looks identical to linear regression

Consider a classification problem.  Adding regularization may cause your classifier to incorrectly classify some training examples (which it had correctly classified when not using regularization, i.e. when λ=0). 

## SVM
### Optimisation Objective
**Alternate view of logistic regression**
$h_{\theta} = \frac{1}{1+e^{-\theta^Tx}}$
If $y = 1$ we want $h_{\theta} \approx 1, \theta^Tx >> 0$
If $y = 0$ we want $h_{\theta} \approx 0, \theta^Tx << 0$

Cost Function for Logistic Regression: 
$$min \theta \frac{1}{m}[\sum_{i=1}^my^{(i)}(-logh_\theta(x^{(i)})) + (1-y^{(i)})((-log(1-h_{\theta}(x^{(i)})))] + \frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$$

**Support Vector Machine**

$$min \theta, C[\sum_{i=1}^my^{(i)}(-logh_\theta(x^{(i)})) + (1-y^{(i)})((-log(1-h_{\theta}(x^{(i)})))] + \frac{1}{2}\sum_{j=1}^n\theta_j^2$$ where $C = \frac{1}{\lambda}$

### Large Margin Intution

$$min \theta, C[\sum_{i=1}^my^{(i)}(-logh_\theta(x^{(i)})) + (1-y^{(i)})((-log(1-h_{\theta}(x^{(i)})))] + \frac{1}{2}\sum_{j=1}^n\theta_j^2$$ where $C = \frac{1}{\lambda}$

If $y = 1$ we want $\theta^Tx \geq 1$
If $y = 0$ we want $\theta^Tx \leq -1$

Decision Boundary
If $C$ is large then cost function is :
$$min \theta,  \frac{1}{2}\sum_{j=1}^n\theta_j^2$$ where $C = \frac{1}{\lambda}$

Whenever $y^{(i)} = 1$: $\theta^Tx^{(i)} \geq 1$
Whenever $y^{(i)} = 0$: $\theta^Tx^{(i)} \leq -1$

<center><img src ="C:\Users\Gurudeep\Desktop\Deep Learning Book\Andre\test31.jpg" alt = "drawing" width="700"/></center><br>


