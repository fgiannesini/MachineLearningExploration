function [newTheta] = derivateCostNeural(theta, layerSize, input, output)
  epsilon = 0.0001;
  grad = zeros(size(theta));
  for layerIndex = 1 : length(layerSize)-1
    firstLayerSize = layerSize(layerIndex);
    secondLayerSize = layerSize(layerIndex + 1);
    for thetaIndex = 1: firstLayerSize * secondLayerSize
      maxTheta = theta;
      maxTheta(layerIndex, thetaIndex) += epsilon;
      maxComputedOutput = computeOutputNeural(input, layerSize, maxTheta);
      maxCout = computeCostNeural(output, maxComputedOutput);

      minTheta = theta;
      minTheta(layerIndex, thetaIndex) -= epsilon;
      minComputedOutput = computeOutputNeural(input, layerSize, minTheta);
      minCout = computeCostNeural(output, minComputedOutput);
      grad(layerIndex,thetaIndex) = (maxCout - minCout) / (2 * epsilon);
    end;
  end;
  newTheta = theta - grad;
end;