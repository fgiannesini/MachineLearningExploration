function [computedOutput] = computeOutputNeural(inputData, layerSize, theta)
  computedOutput = zeros(size(inputData,1), layerSize(length(layerSize)));
  for inputIndex = 1: size(inputData,1)
    result = inputData(inputIndex,:);
    for layerIndex = 1 : length(layerSize)-1
      firstLayerSize = layerSize(layerIndex);
      secondLayerSize = layerSize(layerIndex + 1);
      thetaLayer = reshape(theta(layerIndex, (1:firstLayerSize * secondLayerSize)), firstLayerSize, secondLayerSize);
      newResult = result * thetaLayer;
      newResult = 1./(1 + exp(-newResult)) ;
      result = newResult;
    end;
    computedOutput(inputIndex,:) = result;
  end;
end;