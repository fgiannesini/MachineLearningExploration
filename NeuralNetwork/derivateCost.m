function [newTheta] = derivateCost(theta, layerSize, input, output)
  epsilon = 0.0001;
  grad = zeros(size(theta));
  for layerIndex = 1 : length(layerSize)-1
    firstLayerSize = layerSize(layerIndex) + 1;
    secondLayerSize = layerSize(layerIndex + 1);
    for thetaIndex = 1: firstLayerSize * secondLayerSize

      maxTheta = theta;
      maxTheta(layerIndex, thetaIndex) += epsilon;
      maxComputedOutput = computeOutput(input, layerSize, maxTheta);
      maxCout = computeCost(output, maxComputedOutput);

      minTheta = theta;
      minTheta(layerIndex, thetaIndex) -= epsilon;
      minComputedOutput = computeOutput(input, layerSize, minTheta);
      minCout = computeCost(output, minComputedOutput);
      grad(layerIndex,thetaIndex) = (maxCout - minCout) / (2 * epsilon);
    end;
  end;
  newTheta = theta - grad;
end;