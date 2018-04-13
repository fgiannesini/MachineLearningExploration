clc;
clear all;
close all;

function [computedOutput] = computeOutput(input, layerSize, theta)
  computedOutput = zeros(size(input,1), layerSize(length(layerSize)));
  for inputIndex = 1: size(input,1)
    result = input(inputIndex,:);
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

function [cout] = computeCout(output, computedOutput)
  cout = output.* log(computedOutput) + (1-output) .* log(1-computedOutput);
  cout = - 1 / size(output,1) * sum(cout(:));
end;

%data = load('mnist.mat');

%data.trainX is a (60000,784) matrix which contains the pixel data for training
%data.trainY is a (1,60000) matrix which contains the labels for the training data
%data.testX is a (10000,784) matrix which contains the pixel data for testing
%data.testY is a (1,10000) matrix which contains the labels for the test set*

%X= data.trainX;
%i=reshape(X(1,:),28,28)';
%image(i)

layerSize = [1 2 3];

input = [[1];[2];[3]];
output = [[1,0,0];[0,1,0];[0,0,1]];
theta = [(1:6);(7:12)];

computedOutput = computeOutput(input, layerSize, theta);
cout = computeCout(output, computedOutput);