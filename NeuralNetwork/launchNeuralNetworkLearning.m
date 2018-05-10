function [theta, cout] = launchNeuralNetworkLearning(layerSize, input, output, regularizationCoeff, initialTheta)
  theta = initialTheta;
  
  computedOutput = computeOutput(input, layerSize, theta);
  cout(1) = computeCost(output, computedOutput, theta, regularizationCoeff);

  for coutIndex = 2:1000
##  thetaDerivative= derivateCost(theta, layerSize, input, output, regularizationCoeff);
  
  theta = backPropagate(theta, layerSize, input, output, regularizationCoeff);
##    if abs(sum(sum(thetaDerivative - theta))) > 0.001
##      disp('error on turn');
##      disp(coutIndex);
##      disp(theta);
##      disp(thetaDerivative);
##      break;
##    end;
  
  computedOutput = computeOutput(input, layerSize, theta);  
  cout(coutIndex) = computeCost(output, computedOutput, theta, regularizationCoeff);
    
    if abs(cout(coutIndex) - cout(coutIndex-1)) < 0.0001
      break;
    end;
  end;
end;