function [theta, costData, Xn, Yn]  = applyGradientDescent(X, Y, stepPrecision)

  Xn = normalize(X);
  Yn = normalize(Y);

  theta = rand(size(X,2),1);
  trainingPointCount = size(Xn,2);
  
  for i= 1:1000
    costData(i) = sum((Xn * theta - Yn).^2)/(2*trainingPointCount);
    if length(costData)> 1 && abs(costData(i) - costData(i-1))<0.0001
      break;
    end
    theta = theta - stepPrecision / trainingPointCount * sum((Xn * theta - Yn).* Xn)';
  end

end

