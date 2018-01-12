function [theta, costData, Xn, Yn]  = applyGradientDescent(X, Y, gradientDescentStep, costStabilityDiff, regularizationWeights)

  Xn = normalize(X);
  Yn = normalize(Y);

  theta = rand(size(X,2),1);
  trainingPointCount = size(Xn,2);
  
  for i= 1:1000
    costData(i) = sum((Xn * theta - Yn).^2)/(2*trainingPointCount) + sum(regularizationWeights.*theta.^2);
    if length(costData)> 1 && abs(costData(i) - costData(i-1)) < costStabilityDiff
      break;
    end
    theta = theta .*(1-gradientDescentStep/trainingPointCount * regularizationWeights) - gradientDescentStep / trainingPointCount * sum((Xn * theta - Yn).* Xn)';
  end

end

