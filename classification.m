clear all;
close all;

function [X,Y] = initData(origin, pointsCount, value)
  noise = rand(pointsCount, 2) - 0.5;
  X = ones(pointsCount,2) .* origin + noise;
  Y = ones(pointsCount, 1) * value;
end

pointsCount = 10;

origin1 = [5 1];
[X1,Y1] = initData(origin1, pointsCount, 1);

origin2 = [1 5];
[X2, Y2] = initData(origin2, pointsCount, 0);

X = [X1;X2];
Y = [Y1;Y2];

theta = rand(size(X,2),1);
gradientDescentStep = 0.01;
costStabilityDiff = 0.0001;

for i=1:1000
  estimationFunction = (1/(1+exp(-X * theta)))';
  cost(i) = -1/size(X,1) * sum(Y .* log(estimationFunction) + (1-Y) .* log(1-estimationFunction));
  if length(cost)> 1 && abs(cost(i) - cost(i-1))<costStabilityDiff
    break;
   end
   theta = theta - gradientDescentStep * sum((estimationFunction - Y) .* X)';
end

theta

figure('Position', [200, 200, 1200, 500])
subplot(1,2,1)
plot(X1(:,1),X1(:,2),'x')
hold on;
plot(X2(:,1),X2(:,2),'o')
hold on;
%abscisse = 1:1:max(X,1);
%plot(abscisse, theta' * [abscisse; ones(1,pointsCount)])

subplot(1,2,2)
plot(1:length(cost),cost)