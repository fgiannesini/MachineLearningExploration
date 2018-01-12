  function [Xn] = normalize(X) 
    for i=1:size(X,2)
      diff = (max(X(:,i)) - min(X(:,i)));
      moy = mean(X(:,i));
      if diff != 0
        Xn(:,i) = (X(:,i) - moy) ./ diff;
      else 
        Xn(:,i) = X(:,i);
      end
    end  
  end 