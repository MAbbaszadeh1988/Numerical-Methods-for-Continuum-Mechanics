function [U, x, y] = poisson_2d_direct_kron_compact(f, g, Lx, Ly, nx, ny)
% Compact version - builds operator only for interior points

% Grid
dx = Lx / (nx - 1);
dy = Ly / (ny - 1);
x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);
[X, Y] = meshgrid(x, y);

% Interior dimensions
ni = nx - 2;
nj = ny - 2;

%% Build 1D operators for interior
% 1D Laplacian for interior points (size ni×ni)
ex = ones(ni, 1);
Tx = spdiags([ex, -2*ex, ex], [-1, 0, 1], ni, ni) / dx^2;

% 1D Laplacian for interior points (size nj×nj)
ey = ones(nj, 1);
Ty = spdiags([ey, -2*ey, ey], [-1, 0, 1], nj, nj) / dy^2;

%% 2D operator using Kronecker product
% For interior points only: (I ⊗ T_x + T_y ⊗ I)
I_ni = speye(ni);
I_nj = speye(nj);
A = kron(I_nj, Tx) + kron(Ty, I_ni);

%% Construct RHS with boundary contributions
% Interior source term
F_int = f(X(2:end-1, 2:end-1), Y(2:end-1, 2:end-1));
F = F_int(:);

% Boundary values
U_b = zeros(ny, nx);
U_b(1,:) = g(x, y(1));
U_b(end,:) = g(x, y(end));
U_b(:,1) = g(x(1), y);
U_b(:,end) = g(x(end), y);

% Subtract boundary contributions
% Left boundary
F(1:ni:end) = F(1:ni:end) - U_b(2:end-1, 1)/dx^2;
% Right boundary
F(ni:ni:end) = F(ni:ni:end) - U_b(2:end-1, end)/dx^2;
% Bottom boundary
F(1:ni) = F(1:ni) - U_b(1, 2:end-1)'/dy^2;
% Top boundary
F(end-ni+1:end) = F(end-ni+1:end) - U_b(end, 2:end-1)'/dy^2;

%% Solve
U_int = A \ F;

%% Combine with boundaries
U = U_b;
U(2:end-1, 2:end-1) = reshape(U_int, ni, nj)';

%% Plot
figure;
surf(X, Y, U);
xlabel('x'); ylabel('y'); zlabel('u');
title('Poisson Solution (Kronecker Product - Compact)');
colorbar;

end