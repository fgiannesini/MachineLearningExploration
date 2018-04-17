function [newTheta] = backPropagate(theta,layerSize,inputData, output)
  thetaGrad = zeros(size(theta));
  for inputIndex = 1: size(inputData,1)
    result = inputData(inputIndex,:);
    layerResult = zeros(length(layerSize),max(layerSize));
    layerResult(1,1:length(result)) = result;
    
    for layerIndex = 1 : length(layerSize)-1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      result = 1./(1 + exp(-result * thetaLayer)) ;
      layerResult(layerIndex + 1,1:length(result)) = result;
    end;
    
    layerError = zeros(length(layerSize),max(layerSize));
    layerError(length(layerSize), 1:size(output, 2)) = layerResult(length(layerSize), 1:size(output, 2)) - output(inputIndex,:);
    for layerIndex = length(layerSize)-1:-1:1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      layerError(layerIndex,1:firstLayerSize) = layerError(layerIndex + 1,1:secondLayerSize) * thetaLayer' .* layerResult(layerIndex,1:firstLayerSize).*(1-layerResult(layerIndex,1:firstLayerSize));
      delta = layerError(layerIndex+1,1:secondLayerSize)' * layerResult(layerIndex,1:firstLayerSize);
      thetaGrad(layerIndex,1:firstLayerSize * secondLayerSize) += delta(:)'./size(inputData,1);
    end;
  end;
  newTheta = theta - thetaGrad;
end;