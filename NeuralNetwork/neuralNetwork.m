clc;
clear all;
close all;

%inputSize = 9;
%input = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
%output = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];

  inputSize = 50000;
  hiddenLayerCount = 18; %Tuned

%  crossValidationInput = [[0];[1];[2];[3];[4];[5];[6];[7];[8];[9]];
%  crossValidationOutput = [[1,0,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0,0];[0,0,0,0,1,0,0,0,0,0];[0,0,0,0,0,1,0,0,0,0];[0,0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,0,1]];

  crossValidationSize = 100;  
  crossValidationInput = rand(crossValidationSize * 0.25,1)*9.5;
  crossValidationOutput = generateOutput(crossValidationInput);

  regularizationCoeff = 0;
  
  layerSize = [1 hiddenLayerCount 10];   
  
  theta = initThetaRand(layerSize);

  statIndex = 1;
  statStepSize = inputSize / 300;
for inputIndex = 1:inputSize

  input = rand(1,1)*9.5;
  output = generateOutput(input);
  
  computedOutput = computeOutput(input, layerSize, theta);
  computedCost(inputIndex) = computeCost(output, computedOutput, theta, regularizationCoeff);
  
  theta = backPropagate(theta, layerSize, input, output, regularizationCoeff);

  crossValidationComputedOutput = computeOutput(crossValidationInput, layerSize, theta);  
  crossValidationCost(inputIndex) = computeCost(crossValidationOutput, crossValidationComputedOutput, theta, regularizationCoeff);
    
  if mod(inputIndex,statStepSize) == 0
    costPlot(1,statIndex) = mean(computedCost((statIndex - 1) * statStepSize + 1 : statIndex * statStepSize));  
    costPlot(2,statIndex) = mean(crossValidationCost((statIndex - 1) * statStepSize + 1 : statIndex * statStepSize));  
    [maxValues,maxIndexes] = max(crossValidationComputedOutput');
    crossValidationOutputIndexes = maxIndexes - 1;
    precision(statIndex) = length(find(crossValidationOutputIndexes - round(crossValidationInput)' == 0))/length(crossValidationInput);
    
    statIndex++;
  end;
end;

figure('Position', [300, 300, 1200, 500]);
subplot(1,2,1)
title('Stochastic cost-Cross validation cost');
plot(1:length(costPlot), costPlot(1,:));
hold on;
plot(1:length(costPlot), costPlot(2,:));

subplot(1,2,2)
plot(precision);
title('Précision');