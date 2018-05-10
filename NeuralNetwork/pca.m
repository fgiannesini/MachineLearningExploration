clc;
clear all;
close all;

data = load('mnist.mat');

%data.trainX is a (60000,784) matrix which contains the pixel data for training
%data.trainY is a (1,60000) matrix which contains the labels for the training data
%data.testX is a (10000,784) matrix which contains the pixel data for testing
%data.testY is a (1,10000) matrix which contains the labels for the test set

X = double(data.trainX(1:60000,:));
##i=reshape(X(1,:),28,28)';
##image(i)

##Normalisation
mX = mean(X);
normX = X - mX;

##Calcul de la covariance
cov = normX' * normX / size(normX,1);

##Singular value decomposition
[U,S,V] = svd(cov);

##Recherche de la dimension minimale k conservant 99% de la variance
vectorS = sum(S);
sumS = sum(vectorS);
for k = 1: length(vectorS)
  if sum(vectorS(1:k))/sumS >= 0.99
    break;
  endif
endfor

reducer = U(:,1:k);

##Diminution de la dimension des données d'entrées (k=331 pour les 60000 exemples au lieu de 784) 
input = normX * reducer;

##Retour à la dimension originale
example = input * reducer' + mX;

##figure;
##i2=reshape(example(1,:),28,28)';
##image(i2)