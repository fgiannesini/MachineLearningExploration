function output = generateOutput (input)
  output= zeros(length(input), 10);
  for in = 1:length(input)
      output(in, round(input(in,1))+1) = 1;
  end;
end;