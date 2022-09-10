



```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

#### Transportation Problem
```mermaid
graph TD;
    Transportation_Problem-->Northwest_or_mincost-->uv_method-->know_to_find_&_break_cycles-->Repeat-->uv_method
```

#### TSP
```mermaid
graph TD;
    Travelling_Salesman_Probelm-->Select_2_pairs_of_cities-->replace_edges-->reverse_direction-->find_cost_difference
```

#### Revised Simplex Method
```mermaid
graph TD;
    Revised_Simplex_method-->X_B,Xn=0-->Get_B-->X_B=Bb-1&Xn=0-->Obj=CbB-1b-->Optimality_Check-->Adjacent_BFS-->Repeat-->Optimality_Check
```
#### Integer Linear Programming
```mermaid
graph TD;
    Integer_Linear_Programming-->Knapsack_Problem-->Assignment_Problem-->Branch_and_Bound_Problem
```

#### Knapsack Problem
```mermaid
graph TD;
    Knapsack_Probelm-->a_(Arrange c1/a1>=c2/a2)-->b_(Σ from 1 to k ai<=b)-->c_(assign xi = 1 for all i's)-->d_(calculate k+1th x)-->e_(get z i.e bound)-->f_(solve for variable with help of splitting variable and compute z value)-->g_(again choose splitting variable)-->f_(solve for variable with help of splitting variable and compute z value)
```

#### Assignment Problem
```mermaid
graph TD;
    Assignment_Problem--> a_(row minima and subtract from each row)-->b_(column minima and subtract from each column)-->c_(if N = n Assignment Operation)
    b_(column minima and subtract from each column)-->d_(if N < n get least uncovered element from matrix elements)-->e_(subtract with other uncovered elements and add to elements which are covered twice)-->a_(row minima and subtract from each row)
```

#### Branch and Bound for TSP
```mermaid
graph TD;
    Branch_and_bound_TSP-->a_(given cost matrix row and column reduce it to get LB)-->b_(get max_Θ and that variable becomes branching variable)-->c_(compute lb of new matrix via method 1 or method 2. Choose the one which gives lower LB as TSP's minimisation problem)-->b_(get max_Θ and that variable becomes branching variable)
```
#### Minimization of Non Linear Functions
```mermaid
graph LR;
A[Minimisation of Non Linear Functions] --> B[With no constraint]
    B --> BA[Descent Search] 
    B --> BB[Gradient Descent Method] 
    B --> BC[Steepest Descent Method] 
    B --> BD[Newtons Method and Variations] 
    B --> BE[Conjugate Gradient Method] 
    B --> BF[PSO] 
    B --> BG[ACO] 


A --> C[With Equality Constraint]
    C --> CA[Lagrange Multiplier]
        CA --> CAA[Primal and Dual Feasibility]

    C --> CB[KKT Conditions]
        CB --> CCA[Analytical Method]
        CB --> CCB[Variable Elimination Method]
        CB --> CCD[Solve Dual Problem]
        CB --> CCE[Adapt Techniques]
        CB --> CCG[Unconstrained Optimisation]

A --> D[With Equality and Inequality Contraint]
    D --> DA[KKT Conditions]
        DA --> DAA[Constrained Optimisation]
            DAA --> DAAA[Equality Constraints]
                DAAA --> DAAAA[Primal,Dual Feasibility]
            DAA --> DAAB[InEquality Constraints]
                DAAB --> DAAAB[Primal,Dual,Complementary Slackness] 
    D --> DB[Penalty Method]
    D --> DC[Barrier Method]
        DC --> DCA[Inverse Barrier]
        DC --> DCB[Log Barrier]
    D --> DD[Augmented Lagrangian Method]

```

#### Review on Unconstrained Non Linear Optimisation
```mermaid
graph LR;
A[Review on Unconstrained Non Linear Optimisation] --> B[Gradient Descent Method]
    B --- BA[How to determine lambda^k ?]
        BA --> BAA[Exact Line Search]
        BA --> BAB[Inexact Line Search]
            BAA & BAB --> BAAA[Improvements]
                BAAA --> BAAAA[Approximate Minimisation Rule]
                BAAA --> BAAAB[Armijo Rule]
                BAAA --> BAAAC[Goldstein Rule]
                BAAA --> BAAAD[Limited Minimisation Rule]
                BAAA --> BAAAE[Strong Wolfe Rule]
A --> C[Newtons Method]
A --> D[Conjugate Gradient Method]
    D --- DA[Different ways of Determining beta]
        DA --> DAA[Hestenes-Stiefel Formula ->HS]
        DA --> DAB[Fletcher-Reeves Formula ->FR]
        DA --> DAC[Polak-Ribi ere Formula ->PR]
        DA --> DAD[Polak-Ribi ere Plus Formula->PR+]
        DA --> DAE[Dai-Yuan Formula ->DY]

A --> E[PSO]
A --> F[ACO]
```
#### 
#### Special Techniques for Optimisation in AI
```mermaid
graph TD;
A[Special Techniques for Optimisation in AI] --> B[Stochastic Gradient Method]
A --> C[Non Convex Optimisation Problem]
A --> D[Moreau Envelop Funcion]
    D --> DA[Proximal Operator]
```

#### SVM
```mermaid
graph TD;
A[Classification Problem/ Supervised Learning]-->B[Support Vector Machine]
    B-->BA[Linear Discriminant Function]
        BA --> BAA[Margin]
        BA --> BAB[Support Vectors]
        BAA & BAB--> BAAA[Maximum Margin Linear Classifier->LSVM]
            BAAA --> BAAAA[Maximizing Margin Opti Problem]
                BAAAA --> BAAAAA[Lagrangian Method]
                    BAAAAA ---- BAAAAAB[Apply KKT Conditions]
        BAA & BAB --> BAAB[Soft Margin Classification]
    A --> C[Non Linear SVMS]
```

#### Subgradient
```mermaid
graph TD;
A[Objective Function] --> B[Differentiable]
    B --> BA[Steepest Descent Method]
        BA --- BAA[Uses Approximate Line Search]
A --> C[Non Differentiable]
    C --> CA[Subgradient]
        CA --- CAAA[Uses fixed step length]
        CA --> CAAB[Step size rules]
        CA --> CAAC[Step Size Choices]
        CA --> CAAD[Convergence Analysis]
        CA --> CAAE[Polyak Step Sizes]
```

#### Game Theory
```mermaid
graph TD;
A[Game Theory] --> B[Strategy]
    B --> BA[Pure Strategy]
        BA --- BAA[Saddle Point Exists]
            BAA --> BAAA[Max-Min Strategy -> Primal Problem]
            BAA --> BAAB[Min-Max Startegy -> Dual Problem]
        BA --- BAB[Nash Equilibrium]
    B --> BB[Mixed Strategy]
        BB --- BBA[No Saddle Point]
        BBA --> BBAA[Standard Form of LPP]
```

#### Recommender Systems
```mermaid
graph TD;
A[Approches to Recommender Systems] --> B[Content-Based Recommender Systems->CB]
A --> C[Collaborative Filtering->CF]
    C --> CA[Memory-Based CF]
        CA --> CAA[User-Based CF]
        CA --> CAB[Item-Based CF]
    C --> CB[Model-Based CF]
        CB --> CBA[Matrix Facorization->MF,Clustering,Bayesian Networks]
         CBA --> CBAA[Matrix Factorization]
            CBAA --- CBAAA[Minimise the Loss Error]
                CBAAA --> CBAAAA[Gradient Descent]
                CBAAA --> CBAAAB[Stochastic Gradient Descent]
                CBAAA --> CBAAAC[Mini-Batch GD]
            CBAAAA & CBAAAB & CBAAAC --> CBAAAAA[Evaluation Metrics]

A --> D[Hybrid Recommender Systems]
```
#### Neural Networks
```mermaid
graph TD;
A[Neural Networks] --->B[Multi Layer Perceptron->MLP]
    B -->BA[Error Calculation]
    B -->BB[Back Propagation Algorithm]
        BB --> BBA[Conjugate Gradient Descent Method]
            BBA --- BBAA[Fletcher & Reeves Update Parameters]
            BBA --- BBAB[Polak,Ribiree and Polyak Update Parameters]
        BB --> BBB[Stochastic Gradient Method]
            BBB---BBBA[Randamised Rule]
            BBB ---BBBB[Cyclic Rule]
        BB --> BBC[Mini Batch Gradient Descent Algorithm]
```
#### Case Studies
```mermaid
graph LR;
A[Case Studies] --> B[Finance and Economics]
A --> C[Fraud Detection]
A --> D[Image Segmentation]
A --> E[Dimensionality Reduction]
A --> F[Gene Expression Analysis]
A --> G[Recommender Systems]
A --> H[Image Reconsturction and Robust Face Recognition]
A --> I[Image Denoising]
A --> J[Large Scale Surveillance]
A --> K[Image Construction]
A --> L[Foreground-background seperation]
```
#### Image Processing
```mermaid
graph TD;
A[Image Processing] --> B[Image Denoising as an Optimisation Problem]
    B --- BA[Regularising Term]
    B --- BB[Fidelity Term]
```

#### Others
```mermaid
graph TD;
    Second_Order_Methods

    PSO-->ACO

    Optimisation_of_non_linear_fns-->Lagrange_Multiplier_Method-->Dual_Function-->Equality_Constrained_Convex_Quadractic_Minimisation-->KKT_Conditions

    Algorithms_To_Solve_General_NLP-->Penalty_Method-->Barrier_Method

```