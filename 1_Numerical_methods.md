# Codes for Numerical Methods

## Tutorial Sheet 1
#### Lab Question 1

#### **Bisection Method**

Script File
```matlab
%This file is for executing bisection method bisection.m
global tolerance maxits
tolerance=1e-6;
maxits=30;
a=0;
b=1;
F=@(x) cos(x)-x*exp(x);
class(F);
[rootapprox,status] = bisection(F,a,b);
switch status
    case -1
        disp('Root finding failed')
    case -2
        disp('Initial range does not have the root')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end

%In The following we are calling the bisection.m by using mathfun.m file

global tolerance maxits
tolerance=1e-6;
maxits=30;
a=0;
b=1;
[rootapprox,status] = bisection('mathfun',a,b);
switch status
    case -1
        disp('Root finding failed')
    case -2
        disp('Initial range does not have the root')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end
```
Bisection Method
```matlab
function [rootapprox,status] = bisection(fun,a,b)
%This ia code for bisection method for finding the roots of f(x)=0 with intial 
%approximations a and b 
%status=-1, if maxits reached
%status=-2, if f_a*f_b>0
global tolerance maxits
%maxit-maximum number of iterations
%tolerance
iterations=0;% counter for iteration number
f_a= feval(fun,a);         %f_a= f(a)
f_b= feval(fun,b);          %f_b= f(b)
while(f_a*f_b<0) & iterations<maxits & (b-a)>tolerance
    iterations=iterations+1;
    c=(a+b)/2 ;  %Updated approximation
    f_c=feval(fun,c);  %f_c=f(c)
    if f_c*f_a<0
        b=c; f_b=f_c;
    elseif f_c*f_b<0
        a=c; f_a=f_c;
    else
        rootapprox=c;
    end
end
switch iterations
    case maxits
        status=-1;rootapprox=NaN;
    case 0
        status=-2;rootapprox=NaN;
    otherwise
        status=iterations; rootapprox=c;
end
    
      
        
end
```

#### **Secant Method**

Script File
```matlab
%This is a script file for executing secantfeval.m
global tolerance maxits
tolerance=1e-6;
maxits=30;
a=0;
b=1;
F=@(x) cos(x)-x*exp(x);
[rootapprox,status] = secantfeval(F,a,b);
switch status
    case -1
        disp('Root finding failed')
    case -2
        disp('Initial range does not have the root')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end

%In The following we are calling the secantfeval.m by using mathfun.m file

global tolerance maxits
tolerance=1e-6;
maxits=30;
a=0;
b=1;
[rootapprox,status] = secantfeval('mathfun',a,b);
switch status
    case -1
        disp('Root finding failed')
    case -2
        disp('Initial range does not have the root')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end
```
Secant Method
```matlab
function [rootapprox,status] = secantfeval(fun,a,b)
%This ia code for secant method for finding the roots of f(x)=0 with intial 
%approximations a and b 
%status=-1, if maxits reached
%status=-2, if f_a*f_b>0
global tolerance maxits
%maxit-maximum number of iterations
%tolerance
iterations=0; % counter for iteration number
while iterations<maxits  & abs(b-a)>tolerance
    iterations=iterations+1;
    f_a= feval(fun,a);         %f_a= f(a)
    f_b= feval(fun,b);         %f_b= f(b)
    c=a-f_a*(b-a)/(f_b-f_a);
    a=b;b=c;
end

switch iterations
    case maxits
        status=-1;rootapprox=NaN;
    case 0
        status=-2;rootapprox=NaN;
    otherwise
        status=iterations; rootapprox=c;
end

end
```

#### **Regular Falsi Method**
Script File
```matlab
%This is a script file for executing RegulaFalsi.m
global tolerance maxits
tolerance=1e-1;
maxits=5;
a=0;
b=1;
F=@(x) cos(x)-x*exp(x);
[rootapprox,status] = RegulaFalsi(F,a,b);
switch status
    case -1
        disp('Root finding failed')
    case -2
        disp('Initial range does not have the root')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end
```
Regular Falsi Method
```matlab
function [rootapprox,status] = RegulaFalsi(fun,a,b)
%This ia code for RegulaFalsi-feval method for finding the roots of f(x)=0 with intial 
%approximations a and b 
%status=-1, if maxits reached
%status=-2, if f_a*f_b>0
global tolerance maxits
%maxit-maximum number of iterations
%tolerance
iterations=0; % counter for iteration number
while (iterations<maxits)& (abs(b-a)>tolerance)
    iterations=iterations+1;
    f_a= feval(fun,a);         %f_a= f(a)
    f_b= feval(fun,b);         %f_b= f(b)
    c=b-f_b*((b-a)/(f_b-f_a))
    f_c = feval(fun,c);    %f_c= f(c)
    if f_c*f_a<0
        b=c; 
    elseif f_c*f_b<0
        a=c; 
    else
        rootapprox=c;
    end
    
end

switch iterations
    case maxits
        status=-1;rootapprox=NaN;
    case 0
        status=-2;rootapprox=NaN;
    otherwise
        status=iterations; rootapprox=c;
end
    
        
end
```
#### **Fixed Point Eval**
Script File
```matlab
%This is a script file for executing Fixedpoineval.m
global tolerance maxits
tolerance=1e-3;
maxits=300;
a=0;
global syms x F
F= cos(x)/exp(x);
[rootapprox,status] = Fixedpointeval(F,a);
switch status
    case -1
        disp('Root finding failed')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end
```

**Fixed Point Eval**
```matlab
function [rootapprox,status] = Fixedpointeval(fun,a)
%This ia code for Fixedpoint method using feval
%for finding the roots of x-f(x)=0 with intial 
%approximation a It givs the output 
%status=-1, if maxits reached

global tolerance maxits
global syms x F
%maxit-maximum number of iterations
%tolerance
iterations=0; % counter for iteration number
f_a= eval(subs(fun,x,a))       %f_a= f(a)
%fdiff = diff(fun,x); %Symbolic function, these functions can evaluate by using eval subs.
while iterations<maxits  & abs(a-f_a)>tolerance
    iterations=iterations+1;
    b=f_a;   
    f_a= eval(subs(fun,x,b));         %f_a= f(b)
end

switch iterations
    case maxits
        status=-1;rootapprox=NaN;
    otherwise
        status=iterations; rootapprox=a;
end


end
```

#### **Newton Ralpson Eval**

**Script File**
```matlab
%This is a script file for executing NewtonRaphsoneval.m
global tolerance maxits
tolerance=1e-1;
maxits=30;
a=0;
global syms x F
F= cos(x)-x*exp(x);
[rootapprox,status] = NewtonRaphsoneval(F,a);
switch status
    case -1
        disp('Root finding failed')
    otherwise
        s=sprintf("the approximate root %d found in %d number of iterations",rootapprox, status);
       disp(s)
end
```

**Newton Ralpson Method**
```matlab
function [rootapprox,status] = NewtonRaphsoneval(fun,a)
%This ia code for NewtonRaphson method using feval
%for finding the roots of f(x)=0 with intial 
%approximation a It givs the output 
%status=-1, if maxits reached

global tolerance maxits
global syms x F
%maxit-maximum number of iterations
%tolerance
iterations=0; % counter for iteration number
f_a= eval(subs(fun,x,a));       %f_a= f(a)
fdiff = diff(fun,x); %Symbolic function, these functions can evaluate by using eval subs.
while iterations<maxits  & abs(f_a)>tolerance
    iterations=iterations+1;
    fdiff_a= eval(subs(fdiff,x,a));         %fdiff_a= f'(a)
    b=a-f_a/fdiff_a;
    a=b;
    f_a= eval(subs(fun,x,a));         %f_a= f(a)
end

switch iterations
    case maxits
        status=-1;rootapprox=NaN;
    otherwise
        status=iterations; rootapprox=a;
end
end
```

## Tutorial Sheet 2 
#### **Cholesky Decomposition Script**
```matlab
clc; clear;
A = input("Enter the matrix A: ");
[m,n] = size(A);
if m~=n
    error('Matrix dimensions are not equal; Enter square matrix');
end 
for i = 1:n
    for j = 1:n
        if A(i,j) ~= A(j,i)
            error("Given matrix is not symmetric")
        end 
    end
end 

L = eye(m);
v = zeros(1,m);
D = diag(v);
D(1,1) = A(1,1);   % 1st step in algo


%{
Algorithm
for i = 1,2,...,n do 
di = aii - sum from t=1 to i-1 l_itl_itd_t --> 1
lji = aji - sum from t = 1 to i-1 (ljt lit dt)/di from j = i+ 1 .... n(ith
column of L) --> 2
%}

% GETTING ALL L(j,1) and already got L(1,1) = 1 as L = eye(m)
for j=2:n                % compute the first column of L
    L(j,1) = A(j,1)/D(1,1);   % 2nd step in algo
end 

% L(2,1) = A(2,1)/D(1,1);

% Calculating summations
for i = 2:n
    S1 = 0;
    for t = 1:i-1
        S1 = S1 + L(i,t)^2*D(t,t);    % Summation at 1st eqn
    end 
    D(i,i) = A(i,i) - S1;
    for j = i+1:n
        S2 = 0;
        for t= 1:i-1
            S2 = S2 + L(j,t)*L(i,t)*D(t,t); % Summation at 2nd eqn
        end 
        L(j,i) = (A(j,i) - S2)/D(i,i);       % Other op on 
    end 
end

disp("The matrix A");
disp(A);
disp("The matrix D");
disp(D);
disp("The matrix L");
disp(L);
```

#### **Thomas Decomposition for Tridiagonal Matrix**
```matlab
clc; clear;
T = input("Enter the matrix A: ");
D = input("Enter the matrix D: ");
[m,n] = size(T);
if m~=n
    error('Matrix dimensions are not equal; Enter square matrix');
end 


A = zeros(1,m); B = zeros(1,m); C = zeros(1,m);
X = zeros(m,1);
for i = 1:m-1
    A(i+1) = T(i+1,i);  % Check tridiagonal form to understand it a2 = t(2,1)
    C(i) = T(i,i+1);    % Check tridiagonal form to understand it c1 = t(1,2)
    B(i) = T(i,i);      % Check tridiagonal form to understand it b1 = t(1,1)
end
B(m) = T(m,m);
% Here P and Q are upper triangular elements of a tridiagonal matrix
% P stands for -c' and Q stands for d'  and not c' for simplification of
% code

% Elimination procedure
P = zeros(m,1);
Q = zeros(m,1);
P(1) = -C(1)/B(1);
Q(1) = D(1)/B(1);
i = 2;
while i<=m
    Denominator = B(i) + A(i) * P(i-1);    % Check the eqn used in notes and here its not b - a*c'
    P(i) = -C(i)/Denominator;
    Q(i) = (D(i) - A(i) * Q(i-1))/Denominator;
    i = i + 1;
end 

% Back Substitution
X(m) = Q(m);
for i = m-1:-1:1
    X(i) = P(i)*X(i+1)+Q(i);
end 
disp("The solution vector is")
disp(X)

% disp(A)
% disp(B)
% disp(C)
```

#### **Doolittle Decomposition Script**
```matlab
clc; clear;
A = input("Enter the matrix A: ");
[m,n] = size(A);
if m~=n
    error('Matrix dimensions are not equal; Enter square matrix');
end 

L = eye(m);
U = eye(m);

% Compute first row of U 
for j = 1:n
    U(1,j) = A(1,j);
end 

% Compute first column of L 
for i = 1:n
    L(i,1) = A(i,1)/U(1,1);
end 

% Compute 2,3,4,....nth row of U 
for i = 2:n-1          % i-- index for row number
    for j = i:n      % j-- index for column number
        S = 0;
        for t = 1: i-1   % t--index to compute the sum S
            S = S+ L(i,t)*U(t,j);
        end 
        U(i,j) = A(i,j) - S;
    end 
    
    % Compute the 2,3,4 --nth column of L 
    for k = i:n
        S = 0
        for t = 1:i-1
            S = S + L(k,t)*U(t,i);
        end 
        L(k,i) = (A(k,i) - S)/U(i,i);
    end 
end 

disp("The matrix A");
disp(A);
disp("The matrix L");
disp(L);
disp("The matrix U");
disp(U);
```
