clear all;
close all;

function [X,Y] = initData(origin, pointsCount, noiseAmplitude, value)
  noise = normrnd(zeros(pointsCount,2) + noiseAmplitude,ones(pointsCount, 2)* noiseAmplitude);
  X = [ones(pointsCount,2) .* origin] + noise;
  X = [ones(pointsCount,1) X X(:,1).*X(:,2) X.^2];
  Y = ones(pointsCount, 1) * value;
end

pointsCount = 100;

origin1 = [3 3];
[X1,Y1] = initData(origin1, pointsCount, 1, 1);

origin2 = [3 3];
[X2,Y2] = initData(origin2, pointsCount, 7, 0);
  
regularizationWeights = [0;0;0;300;00;00];

X = [X1;X2];
Y = [Y1;Y2];

X = normalize(X);

theta = rand(size(X,2),1);
gradientDescentStep = 0.01;
costStabilityDiff = 0.0001;

for i=1:1000
  estimationFunction = (1./(1+exp(-X * theta)));
  cost(i) = -1/size(X,1) * sum(Y .* log(estimationFunction) + (1-Y) .* log(1-estimationFunction)) + 1/(2*size(X,1)) * sum(regularizationWeights.*theta.^2);
  if length(cost)> 1 && abs(cost(i) - cost(i-1))<costStabilityDiff
    break;
   end
   theta = theta - gradientDescentStep * sum((estimationFunction - Y) .* X)' - gradientDescentStep/size(X,1) * theta .* regularizationWeights;
end


figure('Position', [200, 200, 1200, 500])
subplot(1,2,1)
zeroIndexes = find(Y == 0);
plot(X(zeroIndexes,2),X(zeroIndexes,3),'x')
hold on;
oneIndexes = find(Y == 1);
plot(X(oneIndexes,2),X(oneIndexes,3),'o')
hold on;
ezplot(@(a,b) theta(1) + a .* theta(2) + b .* theta(3) + a .* b .* theta(4) + a.^2 * theta(5) + b.^2 * theta(6) ,[min(X(:,2)),max(X(:,2)), min(X(:,3)),max(X(:,3))])

subplot(1,2,2)
plot(1:length(cost),cost)