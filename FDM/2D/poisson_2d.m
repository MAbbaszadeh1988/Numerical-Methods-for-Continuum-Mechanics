function [U, x, y] = poisson_2d(f, g, Lx, Ly, nx, ny, max_iter, tol)
% Solves the 2D Poisson equation: ∇²u = f(x,y) on [0,Lx]×[0,Ly]
% with Dirichlet boundary conditions: u = g(x,y) on boundary
%
% Inputs:
%   f - function handle for source term f(x,y)
%   g - function handle for boundary conditions g(x,y)
%   Lx, Ly - domain dimensions
%   nx, ny - number of interior points in x and y directions
%   max_iter - maximum iterations for iterative solver
%   tol - tolerance for convergence
%
% Outputs:
%   U - solution matrix (including boundaries)
%   x, y - coordinate vectors

% Grid spacing
dx = Lx / (nx - 1);
dy = Ly / (ny - 1);
x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);

% Initialize solution matrix
U = zeros(ny, nx);  % Note: U(y,x) indexing

% Apply boundary conditions
for i = 1:nx
    U(1, i) = g(x(i), y(1));      % Bottom boundary
    U(ny, i) = g(x(i), y(ny));     % Top boundary
end
for j = 1:ny
    U(j, 1) = g(x(1), y(j));       % Left boundary
    U(j, nx) = g(x(nx), y(j));      % Right boundary
end

% Jacobi iteration
for iter = 1:max_iter
    U_old = U;
    
    % Update interior points
    for j = 2:ny-1
        for i = 2:nx-1
            U(j,i) = (dy^2*(U_old(j,i+1) + U_old(j,i-1)) + ...
                      dx^2*(U_old(j+1,i) + U_old(j-1,i)) - ...
                      dx^2*dy^2*f(x(i), y(j))) / (2*(dx^2 + dy^2));
        end
    end
    
    % Check convergence
    if norm(U - U_old, 'fro') < tol
        fprintf('Converged after %d iterations\n', iter);
        break;
    end
end

% Plot solution
figure;
surf(x, y, U);
xlabel('x');
ylabel('y');
zlabel('u(x,y)');
title('Solution of Poisson Equation');
colorbar;
end