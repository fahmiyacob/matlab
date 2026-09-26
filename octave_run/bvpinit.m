function solinit = bvpinit (x, yinit, varargin)
%BVPINIT  MATLAB-compatible initial-guess builder for the bvp4c shim.
%   solinit = bvpinit (x, yfun)   yfun a function handle  y = yfun(xi)
%   solinit = bvpinit (x, yvec)   yvec a constant column guess
%   Optional trailing argument -> solinit.parameters (unknown parameters).
  x = x(:).';
  n = numel (x);
  if isa (yinit, 'function_handle')
    y0 = yinit (x(1));
    m  = numel (y0);
    Y  = zeros (m, n);
    Y(:,1) = y0(:);
    for i = 2:n
      yi = yinit (x(i));
      Y(:,i) = yi(:);
    end
  else
    yinit = yinit(:);
    m = numel (yinit);
    Y = repmat (yinit, 1, n);
  end
  solinit = struct ();
  solinit.x = x;
  solinit.y = Y;
  if ~isempty (varargin)
    solinit.parameters = varargin{1};
  end
end
