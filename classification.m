clear all;
close all;

function [X,Y] = initData(origin, pointsCount, value)
  noise = normrnd(zeros(pointsCount,2),ones(pointsCount, 2));
  X = [ones(pointsCount,2) .* origin] + noise;
  X= [ones(pointsCount,1) X];
  Y = ones(pointsCount, 1) * value;
end

pointsCount = 10;

origin1 = [1 5];
[X1,Y1] = initData(origin1, pointsCount, 1);

origin2 = [5 1];
[X2,Y2] = initData(origin2, pointsCount, 0);
  
X = [X1;X2];
Y = [Y1;Y2];

X = normalize(X);

theta = rand(size(X,2),1);
gradientDescentStep = 0.01;
costStabilityDiff = 0.0001;

for i=1:1000
  estimationFunction = (1./(1+exp(-X * theta)));
  preCost = Y .* log(estimationFunction) + (1-Y) .* log(1-estimationFunction);
  cost(i) = -1/size(X,1) * sum(preCost);
  if length(cost)> 1 && abs(cost(i) - cost(i-1))<costStabilityDiff
    break;
   end
   theta = theta - gradientDescentStep * sum((estimationFunction - Y) .* X)';
end

theta

figure('Position', [200, 200, 1200, 500])
subplot(1,2,1)
zeroIndexes = find(Y == 0);
plot(X(zeroIndexes,2),X(zeroIndexes,3),'x')
hold on;
oneIndexes = find(Y == 1);
plot(X(oneIndexes,2),X(oneIndexes,3),'o')
hold on;
%abscisse = min(X(:,2)):0.1:max(X(:,2));
%plot(abscisse, (-theta(2) * abscisse - theta(1))/theta(3))
ezplot(@(a,b) theta(1) + a *theta(2) + b* theta(3),[min(X(:,2)),max(X(:,2)), min(X(:,3)),max(X(:,3))])

subplot(1,2,2)
plot(1:length(cost),cost)