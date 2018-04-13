function [newTheta] = backPropagateNeural(inputData, output, layerSize, theta)
  computedOutput = zeros(size(inputData,1), layerSize(length(layerSize)));
  newTheta = zeros(size(theta));
  for inputIndex = 1: size(inputData,1)
    result = inputData(inputIndex,:);
    layerResult = zeros(length(layerSize),max(layerSize));
    layerResult(1,:) = result;
    
    for layerIndex = 1 : length(layerSize)-1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      newResult = result * thetaLayer;
      layerResult(layerIndex + 1) = newResult;
      result = 1./(1 + exp(-newResult)) ;
    end;
    computedOutput(inputIndex,:) = result;
    
    layerError = zeros(length(layerSize),max(layerSize));
    layerError(length(layerSize),:) = layerResult(length(layerSize),:);
    for layerIndex = length(layerSize)-1:-1:2
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      layerError(layerIndex,:) = thetaLayer * layerError(layerIndex + 1) .* layerResult(layerIndex,:).*(1-layerResult(layerIndex,:));  
      delta = layerError(layerIndex+1,:) * layerResult(layerIndex,:);
      newTheta(layerIndex) += delta(:);
    end;
  end;
  newTheta./size(inputData,1);
end;