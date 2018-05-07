clc;
clear all;
close all;

##Input dans un repere 2D générés aléatoirement
inputSize = 100;
input = [rand(1,inputSize) * 5 ; rand(1,inputSize) * 5];

##itération sur le nombre de Kcenters
for kCount = 2:10
  
  bestCost = Inf;

## Boucle pour tester plusieurs initialisations aléatoires de kcenters
  for iter = 1 : round(inputSize/10)

    kCenters = initKcenters(kCount, input);

    [newKcenters, inputAttribution, lastCost] = computeKmeans(input, kCenters);

    if lastCost< bestCost
      bestCost = lastCost;
      bestKcenters = newKcenters;
      bestInputAttribution = inputAttribution;
    end;
    
  end;
  
  ##Display
  figure;
  title(kCount);
  
  for kCenterIndex = 1:kCount
    kCenterInputIndex = find(bestInputAttribution(1,:) == kCenterIndex);
    hold on;
    plot(input(1,kCenterInputIndex),input(2,kCenterInputIndex),'x');
  end;

  hold on;
  plot(bestKcenters(1,:),bestKcenters(2,:),'o');

  cost(kCount) = bestCost;
end;


figure;
plot(cost);