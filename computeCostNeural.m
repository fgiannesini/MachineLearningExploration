function [cout] = computeCostNeural(output, computedOutput)
  cout = output.* log(computedOutput) + (1-output) .* log(1-computedOutput);
  cout = - 1 / size(output,1) * sum(cout(:));
end;