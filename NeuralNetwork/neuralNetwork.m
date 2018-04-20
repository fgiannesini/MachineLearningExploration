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

  layerSize = [1 18 9];
  input = [[1];[2];[3];[4];[5];[6];[7];[8];[9]];
  output = [[1,0,0,0,0,0,0,0,0];[0,1,0,0,0,0,0,0,0];[0,0,1,0,0,0,0,0,0];[0,0,0,1,0,0,0,0,0];[0,0,0,0,1,0,0,0,0];[0,0,0,0,0,1,0,0,0];[0,0,0,0,0,0,1,0,0];[0,0,0,0,0,0,0,1,0];[0,0,0,0,0,0,0,0,1]];

  theta = initThetaRand(layerSize);

  computedOutput = computeOutput(input, layerSize, theta);
  cout(1) = computeCost(output, computedOutput);

  for coutIndex = 2:5000
  %  thetaDerivative= derivateCost(theta, layerSize, input, output);
    theta = backPropagate(theta, layerSize, input, output);
  %  if abs(sum(sum(thetaDerivative - theta))) > 0.001
  %    disp('error on turn');
  %    disp(coutIndex);
  %    disp(theta);
  %    disp(thetaDerivative);
  %    break;
  %  end;
    
    computedOutput = computeOutput(input, layerSize, theta);
    cout(coutIndex) = computeCost(output, computedOutput);
    
    if abs(cout(coutIndex) - cout(coutIndex-1)) < 0.0001
      break;
    end;
  end;

 [M I] = max(computedOutput')
  
  plot(1:length(cout), cout);