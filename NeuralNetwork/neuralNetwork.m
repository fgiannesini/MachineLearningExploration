clc;
clear all;
close all;

%data = load('mnist.mat');

%data.trainX is a (60000,784) matrix which contains the pixel data for training
%data.trainY is a (1,60000) matrix which contains the labels for the training data
%data.testX is a (10000,784) matrix which contains the pixel data for testing
%data.testY is a (1,10000) matrix which contains the labels for the test set

%X= data.trainX;
%i=reshape(X(1,:),28,28)';
%image(i)


%input = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
%output = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];

inputSize = 30; %Tuned
hiddenLayerCount = 17; %Tuned

input = rand(inputSize,1)*9.5;
output = generateOutput(input);

crossValidationInput = rand(inputSize * 0.25,1)*9.5;
crossValidationOutput = generateOutput(crossValidationInput);

layerSize = [1 hiddenLayerCount 10];  
  
initialTheta = initThetaRand(layerSize);
  
for regularizationCoeff = 0.1:0.1:1;
  
  [theta,learningCost] = launchNeuralNetworkLearning(layerSize, input, output, regularizationCoeff, initialTheta);

  crossValidationComputedOutput = computeOutput(crossValidationInput, layerSize, theta);  
  [maxValues,maxIndexes] = max(crossValidationComputedOutput');
  crossValidationOutputIndexes = maxIndexes - 1;
  precision(hiddenLayerCount) = length(find(crossValidationOutputIndexes - round(crossValidationInput)' == 0))/length(crossValidationInput);
  
  crossValidationCost = computeCost(crossValidationOutput, crossValidationComputedOutput, theta, regularizationCoeff);
  
  costPlot(1,regularizationCoeff * 10) = learningCost(length(learningCost)-1);
  costPlot(2,regularizationCoeff * 10) = crossValidationCost;
end;

plot(1:length(costPlot), costPlot(1,:));
hold on;
plot(1:length(costPlot), costPlot(2,:));

figure;
plot(precision);