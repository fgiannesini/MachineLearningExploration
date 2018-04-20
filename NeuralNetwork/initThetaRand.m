function [theta] = initThetaRand(layerSize)
  maxThetaCount = 0;
  for layerIndex = 1:length(layerSize)-1
    maxThetaCount = max(maxThetaCount, (layerSize(layerIndex)+1) * layerSize(layerIndex +1));
  end;
  theta = zeros(length(layerIndex)-1, maxThetaCount);
  for layerIndex = 1:length(layerSize)-1
    layerCount = (layerSize(layerIndex) + 1) * layerSize(layerIndex +1);
    theta(layerIndex, 1:layerCount) = rand(1,layerCount);
  end;
end;