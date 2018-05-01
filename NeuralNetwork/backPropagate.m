function [newTheta, computedResult] = backPropagate(theta,layerSize,inputData, output, regularizationCoeff)
  thetaGrad = zeros(size(theta));
  maxLayerSize = max(layerSize(1:length(layerSize)));
  for inputIndex = 1: size(inputData,1)
    result = inputData(inputIndex,:);
    layerResult = zeros(length(layerSize), maxLayerSize);
    layerResult(1,1:length(result)) = result;
    
    for layerIndex = 1 : length(layerSize)-1
      result = [1 result];
      firstLayerSize = layerSize(layerIndex) + 1;
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      result = 1./(1 + exp(-result * thetaLayer)) ;
      layerResult(layerIndex + 1,1:length(result)) = result;
    end;
    
    currentLayerError = layerResult(length(layerSize), 1:size(output, 2)) - output(inputIndex,:);
    for layerIndex = length(layerSize)-1:-1:1
      firstLayerSize = layerSize(layerIndex);
      firstLayerSizePlusOne = layerSize(layerIndex) + 1;
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSizePlusOne * secondLayerSize)), firstLayerSizePlusOne, secondLayerSize);
      currentLayerResult = [1 layerResult(layerIndex,1:firstLayerSize)];
      nextLayerError = currentLayerError * thetaLayer' .* currentLayerResult.*(1-currentLayerResult);
      delta = currentLayerResult' * currentLayerError;
      thetaGrad(layerIndex,1:firstLayerSizePlusOne * secondLayerSize) += delta(:)'./size(inputData,1);
      currentLayerError = nextLayerError(2:length(nextLayerError));
    end;
  end;
  
  regularizationMatrix = ones(size(theta)) * regularizationCoeff;
  regularizationMatrix(1:layerSize(1) + 1) = 0;
  newTheta = theta - thetaGrad + regularizationMatrix;
end;