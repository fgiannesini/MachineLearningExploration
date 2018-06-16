clear all;
close all;
clc;

Y = [5 5 0 0; 5 0 0 0; 0 4 0 0 ;0 0 5 4; 0 0 5 0]
Yk = [1 1 1 1;1 0 0 1;0 1 1 0;1 1 1 1; 1 1 1 1]

featureCount = 4;

X = rand(size(Y,1),featureCount);
theta = rand(size(Y,2),featureCount);
regulParam = 0.01;
learningRate = 0.01;

cout(1) = computeCost(X, Y, Yk, theta, regulParam);
 
 for turn = 1:100
   
   for i = 1 : size(X,1)
     for k = 1 : size(X,2)
       coeff = 0;
       for j = 1 : size(theta,1)
         if Yk(i,j) == 1
           coeff += (theta(j,:)* X(i,:)' - Y(i,j)) * theta(j,k);
         endif
       endfor
       coeff += regulParam * X(i,k);
       newX(i,k) = X(i,k) - learningRate * coeff;
     endfor
   endfor
   
   for j = 1 : size(theta,1)
     for k = 1 : size(theta,2)
       coeff = 0;
       for i = 1 : size(X,1)
         if Yk(i,j) == 1
           coeff += (theta(j,:)* X(i,:)' - Y(i,j)) * X(i,k);
         endif
       endfor
       coeff += regulParam * theta(j,k);
       newTheta(j,k) = theta(j,k) - learningRate * coeff;
     endfor
   endfor
   
   X = newX;
   theta = newTheta;
   
   cout(turn +1) = computeCost(X, Y, Yk, theta, regulParam);
endfor

estimate = X * theta'

figure;
plot(cout);
