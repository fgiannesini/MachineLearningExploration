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

layerSize = [1 18 10];
%input = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
%output = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];

%rand(1,20)*9.5
input = [[3.194574];[0.859942];[8.963967];[7.666425];[6.173234];[8.711997];[7.189465];[0.866019];[2.387941];[2.619069];[5.350506];[9.133608];[0.023325];[4.814648];[7.236820];[8.900328];[0.333934];[8.127106];[3.059897];[9.248530]];
output = generateOutput(input);

crossValidationInput = [[3];[4];[3];[9];[2]];

validationInput = [[9];[9];[7];[5];[9]];

regularizationCoeff = 0;
initialTheta = initThetaRand(layerSize);

[theta,cout] = launchNeuralNetworkLearning(layerSize, input, output, regularizationCoeff, initialTheta);

crossValidationComputedOutput = computeOutput(crossValidationInput, layerSize, theta);  
[maxValues,maxIndexes] = max(crossValidationComputedOutput');
crossValidationOutputIndexes = maxIndexes - 1;
precision = length(find(crossValidationOutputIndexes - round(crossValidationInput)' == 0))/length(crossValidationInput);


plot(1:length(cout), cout);