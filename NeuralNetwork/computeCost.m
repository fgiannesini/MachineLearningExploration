function [cout] = computeCost(output, computedOutput,theta, regularizationCoeff)
  cout = output.* log(computedOutput) + (1-output) .* log(1-computedOutput);
  cout = - 1 / size(output,1) * sum(cout(:));
  cout += regularizationCoeff / (2*size(output,1)) * sum(theta(:) .* theta(:));
end;