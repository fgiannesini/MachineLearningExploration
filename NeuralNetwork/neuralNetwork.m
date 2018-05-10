clc;
clear all;
close all;

inputSize = 9;
input = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
output = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];

##  inputSize = 300; %Tuned
  hiddenLayerCount = 18; %Tuned
##
##  input = rand(inputSize,1)*9.5;
##  output = generateOutput(input);

  crossValidationInput = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
  crossValidationOutput = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];
  
##  crossValidationInput = rand(inputSize * 0.25,1)*9.5;
##  crossValidationOutput = generateOutput(crossValidationInput);

  regularizationCoeff = 0;
  cursor = 1;

for  cursor = 1:100

  layerSize = [1 hiddenLayerCount 9];   
  
  initialTheta = initThetaRand(layerSize);

  [theta,learningCost] = launchNeuralNetworkLearning(layerSize, input, output, regularizationCoeff, initialTheta);

  crossValidationComputedOutput = computeOutput(crossValidationInput, layerSize, theta);  
  [maxValues,maxIndexes] = max(crossValidationComputedOutput');
  crossValidationOutputIndexes = maxIndexes - 0;
  precision(cursor) = length(find(crossValidationOutputIndexes - round(crossValidationInput)' == 0))/length(crossValidationInput);
  
  crossValidationCost = computeCost(crossValidationOutput, crossValidationComputedOutput, theta, regularizationCoeff);
  
  costPlot(1,cursor) = learningCost(length(learningCost)-1);
  costPlot(2,cursor) = crossValidationCost;
##  cursor++;
end;

plot(1:length(costPlot), costPlot(1,:));
hold on;
plot(1:length(costPlot), costPlot(2,:));

figure;
plot(precision);
