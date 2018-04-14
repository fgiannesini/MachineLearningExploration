clear all;
close all;

%-- Generate learning set: data are following curve A0 + A1*x +A2*x² + ... + An*x^n with n = length (theorical theta) --
function [X,Y] = generateInputOutput(learningSetCount, theoricalTheta)
  for i=1:length(theoricalTheta)
    X(:,i) = ((1:learningSetCount).^(i-1))';
  end
  
  noise = max(max(abs(X)))/10
  Y= (X + rand(learningSetCount,length(theoricalTheta)) * noise - noise /2) * theoricalTheta;
end;


learningSetCount = 100;
theoricalTheta = rand(5,1) - 0.5;
regularizationWeights = [0;00;00;300;300];

[X,Y] = generateInputOutput(learningSetCount, theoricalTheta);


gradientDescentStep = 0.01
costStabilityDiff = 0.00001
[gradientTheta, costData, gradientX, gradientY] = applyGradientDescent(X,Y,gradientDescentStep, costStabilityDiff, regularizationWeights);

equationTheta = applyEquationResolution(X,Y,regularizationWeights);



%--Display--
figure('Position', [300, 300, 1200, 500]);
abscisse = 1:learningSetCount;

subplot(1,3,1)
plot((1:length(costData)), costData)
title('Fonction Coût Gradient')

subplot(1,3,2)
plot(normalize(abscisse),gradientY,'x','DisplayName','Input');
hold on;
plot(normalize(abscisse),gradientX * gradientTheta, 'DisplayName','Predicted');
legend('show')
title('Gradient Descent')

subplot(1,3,3)
plot(abscisse,Y,'x','DisplayName','Input');
hold on;
plot(abscisse,X * equationTheta, 'DisplayName','Predicted');
legend('show')
title('Equation')