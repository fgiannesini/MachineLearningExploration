function [newTheta] = backPropagate(theta,layerSize,inputData, output)
  computedOutput = zeros(size(inputData,1), layerSize(length(layerSize)));
  newTheta = zeros(size(theta));
  for inputIndex = 1: size(inputData,1)
    result = inputData(inputIndex,:);
    layerResult = zeros(length(layerSize),max(layerSize));
    layerResult(1,1:length(result)) = result;
    
    for layerIndex = 1 : length(layerSize)-1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      newResult = result * thetaLayer;
      layerResult(layerIndex + 1,1:length(newResult)) = newResult;
      result = 1./(1 + exp(-newResult)) ;
    end;
    computedOutput(inputIndex,:) = result;
    
    layerError = zeros(length(layerSize),max(layerSize));
    layerError(length(layerSize),:) = layerResult(length(layerSize),:);
    for layerIndex = length(layerSize)-1:-1:1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      layerError(layerIndex,1:firstLayerSize) = thetaLayer * layerError(layerIndex + 1,1:secondLayerSize)' .* layerResult(layerIndex,1:firstLayerSize)'.*(1-layerResult(layerIndex,1:firstLayerSize))';
      delta = layerError(layerIndex+1,1:secondLayerSize)' * layerResult(layerIndex,1:firstLayerSize);
      newTheta(layerIndex,1:firstLayerSize * secondLayerSize) += delta(:)';
    end;
  end;
  newTheta./size(inputData,1);
end;