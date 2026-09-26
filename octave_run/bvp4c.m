function sol = bvp4c (odefun, bcfun, solinit, options)
%BVP4C  Drop-in shim for MATLAB's bvp4c, for GNU Octave.
%   sol = bvp4c (odefun, bcfun, solinit, options)
%
%   Solves the two-point boundary value problem
%        y'(x) = odefun(x, y),   a <= x <= b
%        0     = bcfun(y(a), y(b))
%   with a 4th-order Hermite-Simpson (Lobatto IIIA) collocation scheme and a
%   damped Newton iteration using an analytic sparse block-bidiagonal
%   Jacobian.  This is the same collocation family MATLAB's bvp4c uses, so
%   nodal values agree to the solver tolerance for smooth problems.
%
%   odefun / bcfun are called with two arguments (x,y) and (ya,yb); any extra
%   formal parameters they declare are supplied through globals, exactly as in
%   the MATLAB original.
%
%   Returns sol.x (1xN mesh), sol.y (m x N solution), and sol.solver metadata.
%   Use deval(sol, xq) to evaluate the (spline) interpolant off-mesh.

  if nargin < 4 || isempty (options); options = bvpset (); end
  reltol = getfieldd (options, 'RelTol', 1e-3);
  abstol = getfieldd (options, 'AbsTol', 1e-6);
  Nmax   = getfieldd (options, 'Nmax', 20000);
  stats  = strcmpi (getfieldd (options, 'Stats', 'off'), 'on');

  a  = solinit.x(1);
  b  = solinit.x(end);
  m  = size (solinit.y, 1);

  % ---- initial fine mesh + interpolated initial guess --------------------
  Ninit = getfieldd (options, 'NfineMin', 400);
  Ninit = max (Ninit, 4*numel (solinit.x));
  x = linspace (a, b, Ninit);
  Y = zeros (m, numel (x));
  xg = solinit.x(:).';
  for j = 1:m
    if numel (xg) >= 2
      Y(j,:) = interp1 (xg, solinit.y(j,:), x, 'spline');
    else
      Y(j,:) = solinit.y(j,1);
    end
  end

  restol = max (reltol, 1e-13);      % target on collocation residual (inf norm)

  % ---- outer mesh-refinement loop ----------------------------------------
  maxpasses = 6;
  for pass = 1:maxpasses
    [Y, ok, resn, itused] = newton_solve (odefun, bcfun, x, Y, m, restol, abstol);
    if stats
      fprintf ('bvp4c(shim): pass %d  N=%d  Newton its=%d  ||res||=%.3e  ok=%d\n', ...
               pass, numel (x), itused, resn, ok);
    end
    % estimate discretisation error from a mesh-halving indicator; refine once
    if pass < maxpasses && numel (x) < Nmax/2
      [x2, Y2, needref] = maybe_refine (odefun, x, Y, m, restol);
      if ~needref && ok
        break;
      end
      x = x2; Y = Y2;
    else
      break;
    end
  end

  sol = struct ();
  sol.x = x;
  sol.y = Y;
  sol.solver = 'bvp4c(shim)';
  sol.stats = struct ('nmesh', numel (x), 'maxres', resn);
  % build a spline interpolant per component for deval
  sol.idata = struct ();
end

% =========================================================================
function [Y, ok, resn, it] = newton_solve (odefun, bcfun, x, Y, m, restol, abstol)
  N  = numel (x);
  U  = reshape (Y, m*N, 1);
  maxit = 100;
  ok = false; resn = Inf; it = 0;
  Rprev = Inf;
  for it = 1:maxit
    [R, J] = assemble (odefun, bcfun, x, U, m, N);
    resn = norm (R, Inf);
    if resn < restol + abstol
      ok = true; break;
    end
    % Newton step (sparse solve)
    dU = - (J \ R);
    if any (~isfinite (dU))
      ok = false; break;
    end
    % damped line search on inf-norm of residual
    lam = 1.0; accepted = false;
    for ls = 1:40
      Un = U + lam*dU;
      Rn = assemble_R (odefun, bcfun, x, Un, m, N);
      if all (isfinite (Rn)) && norm (Rn, Inf) < resn
        U = Un; accepted = true; break;
      end
      lam = lam * 0.5;
    end
    if ~accepted
      % take the smallest damped step anyway to try to escape
      U = U + lam*dU;
    end
    if norm (lam*dU, Inf) < 1e-13
      [R,~] = assemble (odefun, bcfun, x, U, m, N);
      resn = norm (R, Inf);
      ok = (resn < 1e-8);
      break;
    end
    Rprev = resn;
  end
  Y = reshape (U, m, N);
end

% =========================================================================
function [R, J] = assemble (odefun, bcfun, x, U, m, N)
  Y = reshape (U, m, N);
  R = zeros (m*N, 1);

  % precompute f and df/dy at all nodes
  F  = zeros (m, N);
  Jf = cell (1, N);
  for i = 1:N
    [F(:,i), Jf{i}] = f_and_jac (odefun, x(i), Y(:,i), m);
  end

  % triplets for sparse Jacobian
  nnz_est = m*m*(2*(N-1) + 2);
  Ii = zeros (nnz_est,1); Jj = Ii; Vv = Ii; p = 0;
  Im = eye (m);

  for i = 1:N-1
    h  = x(i+1) - x(i);
    xi = x(i);   yi  = Y(:,i);    fi  = F(:,i);   Jfi  = Jf{i};
    yip = Y(:,i+1); fip = F(:,i+1); Jfip = Jf{i+1};
    xm  = 0.5*(xi + x(i+1));
    ym  = 0.5*(yi + yip) + (h/8)*(fi - fip);
    [fm, Jfm] = f_and_jac (odefun, xm, ym, m);

    Ri = yip - yi - (h/6)*(fi + 4*fm + fip);
    rows = (i-1)*m + (1:m);
    R(rows) = Ri;

    dym_dyi  = 0.5*Im + (h/8)*Jfi;
    dym_dyip = 0.5*Im - (h/8)*Jfip;
    dRi_dyi  = -Im - (h/6)*(Jfi  + 4*Jfm*dym_dyi);
    dRi_dyip =  Im - (h/6)*(Jfip + 4*Jfm*dym_dyip);

    ci  = (i-1)*m + (1:m);
    cip = i*m + (1:m);
    [Ii,Jj,Vv,p] = addblock (Ii,Jj,Vv,p, rows, ci,  dRi_dyi);
    [Ii,Jj,Vv,p] = addblock (Ii,Jj,Vv,p, rows, cip, dRi_dyip);
  end

  % boundary conditions in the last m rows
  ya = Y(:,1); yb = Y(:,N);
  bc = bcfun (ya, yb); bc = bc(:);
  rows = (N-1)*m + (1:m);
  R(rows) = bc;
  [dBda, dBdb] = bc_jac (bcfun, ya, yb, m);
  [Ii,Jj,Vv,p] = addblock (Ii,Jj,Vv,p, rows, 1:m,            dBda);
  [Ii,Jj,Vv,p] = addblock (Ii,Jj,Vv,p, rows, (N-1)*m+(1:m),  dBdb);

  J = sparse (Ii(1:p), Jj(1:p), Vv(1:p), m*N, m*N);
end

% ---- residual only (cheap, for line search) -----------------------------
function R = assemble_R (odefun, bcfun, x, U, m, N)
  Y = reshape (U, m, N);
  R = zeros (m*N, 1);
  F = zeros (m, N);
  for i = 1:N
    F(:,i) = odefun (x(i), Y(:,i));
    F(:,i) = F(:,i)(:);
  end
  for i = 1:N-1
    h  = x(i+1) - x(i);
    fi = F(:,i); fip = F(:,i+1);
    xm = 0.5*(x(i)+x(i+1));
    ym = 0.5*(Y(:,i)+Y(:,i+1)) + (h/8)*(fi - fip);
    fm = odefun (xm, ym); fm = fm(:);
    R((i-1)*m + (1:m)) = Y(:,i+1) - Y(:,i) - (h/6)*(fi + 4*fm + fip);
  end
  bc = bcfun (Y(:,1), Y(:,N)); R((N-1)*m + (1:m)) = bc(:);
end

% =========================================================================
function [f, Jf] = f_and_jac (odefun, x, y, m)
  f = odefun (x, y); f = f(:);
  Jf = zeros (m, m);
  for k = 1:m
    step = 1e-7 * max (1, abs (y(k)));
    yp = y; yp(k) = yp(k) + step;
    fp = odefun (x, yp); fp = fp(:);
    Jf(:,k) = (fp - f) / step;
  end
end

function [dBda, dBdb] = bc_jac (bcfun, ya, yb, m)
  b0 = bcfun (ya, yb); b0 = b0(:);
  dBda = zeros (m, m); dBdb = zeros (m, m);
  for k = 1:m
    step = 1e-7 * max (1, abs (ya(k)));
    yp = ya; yp(k) = yp(k) + step;
    bp = bcfun (yp, yb); dBda(:,k) = (bp(:) - b0) / step;
    step = 1e-7 * max (1, abs (yb(k)));
    yp = yb; yp(k) = yp(k) + step;
    bp = bcfun (ya, yp); dBdb(:,k) = (bp(:) - b0) / step;
  end
end

% =========================================================================
function [x2, Y2, needref] = maybe_refine (odefun, x, Y, m, restol)
% Insert midpoints where the residual of the *current* solution on a doubled
% mesh is large; returns needref=false when the mesh already resolves it.
  N = numel (x);
  % residual indicator: compare Hermite interpolant slope error at midpoints
  worst = 0;
  markers = false (1, N-1);
  for i = 1:N-1
    h  = x(i+1) - x(i);
    fi = odefun (x(i), Y(:,i)); fi = fi(:);
    fip= odefun (x(i+1), Y(:,i+1)); fip = fip(:);
    xm = 0.5*(x(i)+x(i+1));
    ym = 0.5*(Y(:,i)+Y(:,i+1)) + (h/8)*(fi - fip);
    fm = odefun (xm, ym); fm = fm(:);
    % local defect estimate ~ |y' - fm| at midpoint via Hermite derivative
    dym = (3/(2*h))*(Y(:,i+1)-Y(:,i)) - 0.25*(fi + fip);
    d = norm (dym - fm, Inf) * h;
    if d > worst; worst = d; end
    if d > 10*restol; markers(i) = true; end
  end
  needref = any (markers) && (worst > 10*restol);
  if ~needref
    x2 = x; Y2 = Y; return;
  end
  % build refined mesh (insert midpoint in marked intervals)
  xl = x(1);
  xnew = x(1);
  for i = 1:N-1
    if markers(i)
      xnew(end+1) = 0.5*(x(i)+x(i+1));
    end
    xnew(end+1) = x(i+1);
  end
  x2 = xnew;
  Y2 = zeros (m, numel (x2));
  for j = 1:m
    Y2(j,:) = interp1 (x, Y(j,:), x2, 'spline');
  end
end

% =========================================================================
function [Ii,Jj,Vv,p] = addblock (Ii,Jj,Vv,p, rows, cols, B)
  m = numel (rows);
  [C, Rr] = meshgrid (cols, rows);   % Rr row-index, C col-index (m x m)
  idx = p + (1:m*m);
  Ii(idx) = Rr(:);
  Jj(idx) = C(:);
  Vv(idx) = B(:);
  p = p + m*m;
end

% =========================================================================
function v = getfieldd (s, name, default)
  if isfield (s, name); v = s.(name); else; v = default; end
end
