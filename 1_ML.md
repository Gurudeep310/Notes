# ML Paradigms

```mermaid
graph LR;
A[ML Paradigms] --> AA[Supervised Learning]
    AA --> AAA[Regression]
        AAA --> AAAA[Methods to make model]
            AAAA --> AAAAA[Linear Regression]
            AAAA --> AAAAB[Polynomial Regression]
            AAAA --> AAAAC[Locally Weighted Linear Regression]
            AAAA --> AAAAD[Decision Trees]
            AAAA --> AAAAE[K Nearest Neighbours]
            AAAA --> AAAAF[SVM]
            AAAA --> AAAAG[Ensemble Learning]
            AAAA --> AAAAH[Random Forest]
    AA --> AAB[Classification]
        AAB --> AABA[Binary Classification]
            AABA --> AABAA[Linear Regression for Classification]
            AABA --> AABAB[Logistic Regression]
            AABA --> AABAC[K Nearest Neighbours]
            AABA --> AABAD[SVM -> can do both linear and non linear classification]
        AAB --> AABB[Multi Class Classification]
            AABB --> AABBA[Linear Regression for Classification using Strategies]
            AABB --> AABBB[Decision Trees]
            AABB --> AABBC[Gaussian Naive Bayes Theorem]
            AABB --> AABBD[Ensemble Learning]
            AABB --> AABBE[Random Forest]

A --> AB[Unsupervised Learning]
    AB --> ABA[Rule Mining]
    AB --> ABB[Clustering]
    AB --> ABC[Feature Selection]
    AB --> ABD[Collaborative Filtering]
A --> AC[Reinforcement Learning]

````

# Features

```mermaid
graph TD;
B[Features] --> BA[Numeric]
B --> BB[Strings]
    BB --> BBA[Nominal]
    BB --> BBB[Ordinal]

C[Distance] --> CA[Manhattan Distance]
C --> CB[Euclidean Distance]
C --> CC[Minkowski's Distance]
```
# Errors

```mermaid
graph TD
E[Errors] --> EA[Sum of Sqaured Errors]
E --> EB[Mean squared Error]
E --> EC[Least Square Error]
E --> ED[Mean Absolute Error]
```
# Linear Regression

```mermaid
graph LR;
    AAAAA[Linear Regression]
        AAAAA --> AAAAAA[Univariate Regression]
            AAAAAA --- AAAAAAA[Loss Function]
        AAAAA --> AAAAAB[Multivariate Linear Regression]
            AAAAAB --- AAAAABA[Loss Function]
        AAAAA --> AAAAAC[Single Output Regression]
            AAAAAC --- AAAAACA[Loss Function]
        AAAAA --> AAAAAD[Multi Output Regression]
            AAAAAD --- AAAAADA[Loss Function]
        AAAAA --> AAAAAE[Determining parameters / Min Loss Function]
            AAAAAE --> AAAAAEA[Solve Linear eqns]
            AAAAAE --> AAAAAEB[Closed Form Soln]
                AAAAAEB --> AAAAAEBA[Overfitting] --- AAAAAEBAA[Regularising the error/loss fn]
                    AAAAAEBAA --> AAAAAEBAAA[L1 Regularisation]
                    AAAAAEBAA --> AAAAAEBAAB[L2 Regularisation]
            AAAAAE --> AAAAAEC[Gradient Descent Method/Alternating weight optimisation]
                AAAAAEC --> AAAAAECA[Feature Scaling]
                    AAAAAECA --> AAAAAECAA[Normalisation]
                    AAAAAECA --> AAAAAECAB[Standardisation]
        AAAAA --> AAAAAF[Eager Algorithm]

    A[Linear Regression For Classification] --> AA[Binary Class Classification]
    A --> AB[Multi Class Classification]
        AB --- ABA[Convert Multi Class to Binary Class dataset via]
            ABA --> ABAA[One vs One]
                ABAA --- ABAAA[Info Loss: N-2N/k]
            ABA --> ABAB[One vs All]
                ABAB --- ABABA["Info Loss: (N/k)/(N-N/K)"]
            ABA --> ABAC["Error Correcting Output Code(best)"]
```
# KNN

```mermaid
graph LR
 A[K Nearest Neighbours For Regression] -->AA[Determining Parameter k]---AAA[M-fold Cross Validation-get best k-]
    AAA --> AAAA[min of sum of SAE over columns]
    AAA --> AAAB[min of sum of SAE over rows]
 A --> AB["Loss Function"]
 A --> AC[Not good when data is high dimensional]
 A --> AD[Lazy Algorithm]

 B[K Nearest Neighbours For Classification] --> BA[M-fold Cross Validation] --> BAA[Accuracy]
```

```mermaid
graph LR;
A[ML Algorithm Component] --> AA[Representation] --> AB["Optimisation(Mostly in eager algo's)"] --> AC[Evaluation]
```
# Logistic Regression

```mermaid
graph LR;
A[Logistic Regression] --> AA[Loss Function]
A --> AB[Determining Parameters/Min Loss fn] --- AC[Gradient Descent Method]
```
Normalisation;Standardisation(For Linear Regression);Correlation

# Decision Trees

```mermaid
graph LR;
A[Decision Trees for Classification] --> AA[How to Measure Uncertainity ?]
    AA --- AAA[Shannos Entropy] 
        AAA --- AAAA[Information Gain -> 1-Entropy]
        AAA -.- AAAB[Feature to be used for classification is the one which gives least entropy or which provides max info gain]
    A --> AB[ID3 Algorithm] 
    A --> AC[Overfitting]
        AC --> ACA[Pruning Strategies]
            ACA --> ACAA[Pre Pruning Strategies]
                ACAA --> ACAAA[Limiting Depth]
                ACAA --> ACAAB[Limiting Leaves]
                ACAA --> ACAAC[Condition on Entropy]
                ACAA --> ACAAD[Condition of Probability Score]
                ACAA --> ACAAE[Control Greediness]
                ACAA --> ACAAF[Condition on Info Gain]
                ACAA --> ACAAE[Condition on Number of Samples]

        AC --> ACB[Post Pruning Strategies]
                ACB --> ACBA["Subtree Error Pruning (best)"]


B[Decision Trees for Regression] --> BA[How to Select a Threshold to split?] 
    BA --> BAA[Variance Sum]
        BAA -.- BAAA[Select a feature which has least var sum]
    B --> BB[Overfitting]
        BB --> ACA & ACB
```
https://www.analyticsvidhya.com/blog/2020/06/4-ways-split-decision-tree/

# SVM

```mermaid
graph LR;
A[SVM for Classification] --> AA[Max/Hard Margin]
A --> AB[Soft Margin]
    AA & AB --- ABA[Lagrange Multipliers to solve]

B[SVM for Regression]
```

# Random Forest
```mermaid
graph LR;
A[Ensemble Learning]-.-AA[Multiple models trained and unification of prediction]
A -.- AB[Model Should be a weak learner]
    AB --> ABA[Create Different training sets]
        ABA --> ABAA[Bagging]
            ABAA --> ABAAA[Instance Bagging]
            ABAA --> ABAAB[Feature Bagging]
            ABAA --> ABAAC[Voting]
        ABA --> ABAB[Boosting]
            ABAB --> ABABA[Ada Boost]
                ABABA -.- ABABAA[We do not grow the tree fully] 
                ABABA --- ABABAB[Quantity] 
                    ABABAB --- ABABABA[Normalization]
                ABABA --> ABABAC[Instead of Instance Bagging apply Cumulative Distribution]
                ABABA --> ABABAD[Prediction -> Majority Voting]
        ABA --> ABAC[Stacking]
            ABAC -.- ABACA[Models made heterogenous]
            ABAC --> ABACB[Accuracy Metric-> MAE]
            ABAC -.- ABACC[Suffers from overfitting]
                ABACC --> ABACCA[Blending]
            ABAC --> ABACD[Prediction Function]
                ABACD --> ABACDA[How to Determine wi ? Linear Weightage]
                    ABACDA --> ABACDAA[Weighted Linear Regression]
                ABACD --> ABACDB[Hierachial Mixture of experts]
A --> AC[Unification of Prediction]
    AC --> ACA[For Classification]
        ACA --> ACAA[Majority Vote]
        ACA --> ACAB[Border Count]
    AC --> ACB[For Regression]
```

# Gaussian Naive Bayes Theorem

```mermaid
graph LR;
A[Gaussian Naive Bayes Theorem] --> AA[For Categorical Features]
A --> AB[For Numerical Features]
    AB --> ABA[Calculate Probability using Gaussion distribution PDF]
A --> AC[Terms to be Known]
    AC --> ACA["Posterior Probability P(A|B)"]
    AC --> ACB["Priori Probability P(A|B)"]
    AC --> ACC["Probability of Evidence P(B)"]
    AC -.- ACD["If P(B) is small use Laplace Correction"]
```

# Accuracy Metrics
```mermaid
graph TD;
A[Accuracy Metric] --> AA[Residual Error]
A --> AB[R Sqaure Metric]
A --> AC[Accuracy]
A --> AD[Precision]
A --> AE[Recall]
A --> AF[F1 Score]
A --> AG[AUC ROC]
```

# Deep Learning
```mermaid
graph LR;
A[Hebbian Learning]
B[Competitive Learning]
C[Associative Memory] --> CA[Static]
C --> CB[Dynamic]
    CA & CB --> CBA[Auto Associative]
        CBA --> CBAA[Discrete Hopfield Network]
    CA & CB --> CBB[Heteroassociative memory]
        CBB --> CBBA[Bidirectional Associative Memory]

```