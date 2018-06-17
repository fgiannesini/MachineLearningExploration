clear all;
close all;
clc;


R =...
[-1 -1 -1 -1 0 -1;...
 -1 -1 -1 0 -1 100;...
 -1 -1 -1 0 -1 -1;...
 -1 0 0 -1 0 -1;...
 0 -1 -1 0 -1 100;...
 -1 0 -4 -4 0 100];
 
 Q = zeros(size(R));
 learningRate = 0.1;
 for turn = 1:100
   initialState = ceil(rand(1) * size(R,2));
   possibleStates = find(R(initialState,:) > -1);
   nextState= possibleStates(ceil(rand(1) * length(possibleStates)));
   bestReward = max(Q(nextState,:));
   Q(initialState, nextState) = (1- learningRate) * Q(initialState, nextState) ...
   + learningRate * (R(initialState, nextState) + bestReward);
 endfor
 
Q = Q / max(abs(Q(:))) * 100;

states = zeros(size(R));
for state = 1: size(R,1)
  states(state,1) = state - 1;
  currentState = state;
  move = 2;
  while currentState != size(R,1)
    [nextStateReward, nextStateIndex] = max(Q(currentState, :));
    states(state,move++) = nextStateIndex - 1;
    currentState = nextStateIndex;
  endwhile
endfor

Q
states
