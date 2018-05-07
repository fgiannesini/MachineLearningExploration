function [kCenters] = initKcenters(kCount,input)
  inputSize = size(input,2);
  kCountIndexes = ceil(rand(kCount,1) * inputSize); 
  kCenters = zeros(2,length(kCountIndexes));
  for kCenterIndex=1:kCount
    kCenters(1,kCenterIndex) = input(1,kCountIndexes(kCenterIndex));
    kCenters(2,kCenterIndex) = input(2,kCountIndexes(kCenterIndex));
  end;
endfunction
