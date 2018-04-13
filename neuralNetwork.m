clc;
clear all;
close all;

data = load('mnist.mat');

%data.trainX is a (60000,784) matrix which contains the pixel data for training
%data.trainY is a (1,60000) matrix which contains the labels for the training data
%data.testX is a (10000,784) matrix which contains the pixel data for testing
%data.testY is a (1,10000) matrix which contains the labels for the test set

X= data.trainX;
i=reshape(X(1,:),28,28)';
image(i)

layerSize = [1 6 3];
input = [[1];[2];[3]];
output = [[1,0,0];[0,1,0];[0,0,1]];

maxThetaCount = 0;
for layerIndex = 1:length(layerSize)-1
  maxThetaCount = max(maxThetaCount, layerSize(layerIndex) * layerSize(layerIndex +1));
end;
theta = zeros(length(layerSize)-1, maxThetaCount);
for layerIndex = 1:length(layerSize)-1
  layerCount = layerSize(layerIndex) * layerSize(layerIndex +1);
  theta(layerIndex, 1:layerCount) = rand(1,layerCount);
end;

computedOutput = computeOutputNeural(input, layerSize, theta);
cout(1) = computeCostNeural(output, computedOutput);
for coutIndex=2:2000
  theta = derivateCostNeural(theta, layerSize, input, output);
  computedOutput = computeOutputNeural(input, layerSize, theta);
  cout(coutIndex) = computeCostNeural(output, computedOutput);
%  if abs(cout(coutIndex) - cout(coutIndex-1)) < 0.0001
%    break;
%  end;
end;

plot((1:length(cout)), cout);