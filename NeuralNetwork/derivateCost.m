function [newTheta] = derivateCost(theta, layerSize, input, output, regularizationCoeff)
  epsilon = 0.0001;
  grad = zeros(size(theta));
  for layerIndex = 1 : length(layerSize)-1
    firstLayerSize = layerSize(layerIndex) + 1;
    secondLayerSize = layerSize(layerIndex + 1);
    for thetaIndex = 1: firstLayerSize * secondLayerSize

      maxTheta = theta;
      maxTheta(layerIndex, thetaIndex) += epsilon;
      maxComputedOutput = computeOutput(input, layerSize, maxTheta);
      maxCout = computeCost(output, maxComputedOutput,theta, regularizationCoeff);

      minTheta = theta;
      minTheta(layerIndex, thetaIndex) -= epsilon;
      minComputedOutput = computeOutput(input, layerSize, minTheta);
      minCout = computeCost(output, minComputedOutput, theta, regularizationCoeff);
      grad(layerIndex,thetaIndex) = (maxCout - minCout) / (2 * epsilon);
    end;
  end;
  regularizationMatrix = ones(size(theta)) * regularizationCoeff ./ size(output,1);
  regularizationMatrix(1:layerSize(1) + 1) = 0;
  newTheta = theta - grad + regularizationMatrix;
end;