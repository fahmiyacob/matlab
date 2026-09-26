function opts = bvpset (varargin)
%BVPSET  Minimal MATLAB-compatible options builder for the bvp4c shim.
%   Recognises RelTol, AbsTol, Stats, Nmax, FJacobian, BCJacobian, Vectorized.
%   Unknown names are stored verbatim (case preserved).
  opts = struct ('RelTol', 1e-3, 'AbsTol', 1e-6, 'Stats', 'off', ...
                 'Nmax', 20000, 'NfineMin', 400);
  for k = 1:2:numel (varargin)
    name = varargin{k};
    val  = varargin{k+1};
    switch lower (name)
      case 'reltol',    opts.RelTol = val;
      case 'abstol',    opts.AbsTol = val;
      case 'stats',     opts.Stats  = val;
      case 'nmax',      opts.Nmax   = val;
      otherwise,        opts.(name) = val;   % keep anything else
    end
  end
end
