function [theta] =  applyEquationResolution(X,Y,regularizationWeights)

regularizationMatrix = eye(length(regularizationWeights)).* regularizationWeights * max(max(X))^2;

theta = pinv(X'*X + regularizationMatrix)*X'*Y;
end;
