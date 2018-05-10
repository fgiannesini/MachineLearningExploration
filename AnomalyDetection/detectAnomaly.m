clc;
clear all;
close all;

pkg load nan;

##Generate normal and anormal inputs
nSize = 100;
nInput = ones(nSize,2) * 5 + (rand(nSize,2) - 0.5) * 6;  

anInput = [2 2; 8 8; 2 8; 8 2];

##Compute one gaussian for each features on normal input
moy = mean(nInput);
ecartType = sqrt(var(nInput,1));

nNorm = normpdf(nInput,moy,ecartType);

##Compute probability for normal inputs
nProb = prod(nNorm');

##Determine limit probability for normal/anormal input
epsilon = min(nProb);

##Compute anormal inputs probability
anNorm = normpdf(anInput,moy,ecartType);
anProb = prod(anNorm');

##Check none anormal input is above limit 
if length(find(anProb >= epsilon)) > 0
  disp('Espilon too high');
end

figure;
title('Features');
plot(nInput(:,1),nInput(:,2),'x');
hold on;
plot(anInput(:,1),anInput(:,2),'o');

figure;
title('Probability');
plot(nProb,'x');
hold on;
plot(anProb,'o');