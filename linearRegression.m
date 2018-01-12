clear all;
close all;

function [X,Y] = generateInputOutput(inputCount, theoricalTheta)
  X= [(1:inputCount)'.^2, (1:inputCount)', ones(1, inputCount)'];
  noise = max(max(abs(X)))/10
  Y= (X + rand(inputCount,size(theoricalTheta,1)) * noise - noise /2) * theoricalTheta;
end;

inputCount = 100;
theoricalTheta = [1;2;3];
[X,Y] = generateInputOutput(inputCount, theoricalTheta);

precisionStep = 0.01
[gradientTheta, costData, gradientX, gradientY] = applyGradientDescent(X,Y,precisionStep);

equationTheta = applyEquationRegression(X,Y);

figure('Position', [300, 300, 1200, 500]);

subplot(1,3,1)
plot((1:length(costData)), costData)
title('Fonction Coût Gradient')

subplot(1,3,2)
plot(gradientX(:,1),gradientY,'x','DisplayName','Input');
hold on;
plot(gradientX(:,1),gradientX * gradientTheta, 'DisplayName','Predicted');
legend('show')
title('Gradient Descent')

subplot(1,3,3)
plot(X(:,1),Y,'x','DisplayName','Input');
hold on;
plot(X(:,1),X * equationTheta, 'DisplayName','Predicted');
legend('show')
title('Equation')