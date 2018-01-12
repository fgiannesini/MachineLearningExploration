function [theta] =  applyEquationRegression(X,Y)
theta = pinv(X'*X)*X'*Y;
end;
