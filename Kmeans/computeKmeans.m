function [kCenters, inputAttribution, lastCost] = computeKmeans(input, kCenters)
  kCountSize = size(kCenters,2);
  inputSize = size(input,2);
  inputAttribution = zeros(2,inputSize);
  for iter = 1:100
  newKCenters = zeros(3,kCountSize);
    for inputIndex = 1:inputSize
      distanceX = input(1,inputIndex) - kCenters(1,:);
      distanceY = input(2,inputIndex) - kCenters(2,:);
      distance = distanceX.*distanceX + distanceY.*distanceY;
      [minValue,minIndex] = min(distance);
      newKCenters(1,minIndex) += input(1,inputIndex);
      newKCenters(2,minIndex) += input(2,inputIndex);
      newKCenters(3,minIndex) ++;
      inputAttribution(1,inputIndex) = minIndex;
      inputAttribution(2,inputIndex) = minValue;
    end;

  newKCenters(1:2,:) = newKCenters(1:2,:) ./ newKCenters(3,:);

  kCenters = newKCenters(1:2,:);

  cost(iter) = sum(inputAttribution(2,:)) / inputSize;
  if iter >=2 && abs(cost(iter) - cost(iter-1)) < 0.01
    break;
  end;
end;

lastCost = cost(iter);

endfunction
