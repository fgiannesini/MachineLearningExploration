function [cout] = computeCost(X, Y, Yk, theta,regulParam)
   cout = 0.5 * sum(sum(((theta * X')' - Y).* Yk).^2)...
 + regulParam * 0.5 * sum(X(:).^2) + regulParam * 0.5 * sum(theta(:).^2);
endfunction
