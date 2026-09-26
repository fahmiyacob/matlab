function Y = deval (sol, xq)
%DEVAL  Evaluate a bvp4c-shim solution at points xq (spline interpolation).
%   Y = deval (sol, xq)  returns an (m x numel(xq)) matrix.
  xq = xq(:).';
  m  = size (sol.y, 1);
  Y  = zeros (m, numel (xq));
  for j = 1:m
    Y(j,:) = interp1 (sol.x, sol.y(j,:), xq, 'spline');
  end
end
