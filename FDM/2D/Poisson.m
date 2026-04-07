%%=========================================================
% 2D Poisson Problem:  -Delta u = f  on (0,1)x(0,1)
% Zero Dirichlet boundary conditions
% 5-point finite difference method
%%=========================================================

clear; 
clc; 
close all;

%% Number of interior grid points
N = 50;                  
h = 1/(N+1);             

%% Grid
x = linspace(0,1,N+2);
y = linspace(0,1,N+2);
xi = x(2:end-1);
yi = y(2:end-1);

%% Right-hand side
f = @(x,y) 2*pi^2*sin(pi*x).*sin(pi*y);

%% Exact solution (for verification)
u_exact = @(x,y) sin(pi*x).*sin(pi*y);

%---------------------------------------------------------
% Build RHS vector
%---------------------------------------------------------
F = zeros(N,N);
for i = 1:N
    for j = 1:N
        F(i,j) = f(xi(i), yi(j));
    end
end

F = reshape(F, N^2, 1);

%---------------------------------------------------------
% Construct coefficient matrix A for -Delta
%---------------------------------------------------------
e = ones(N,1);

% 1D second difference matrix for -d^2/dx^2
T = spdiags([-e 2*e -e], -1:1, N, N);

I = speye(N);

% 2D Laplacian using Kronecker products
A = (kron(I,T) + kron(T,I)) / h^2;

%---------------------------------------------------------
% Solve linear system
%---------------------------------------------------------
U = A \ F;

% Reshape to grid
U = reshape(U, N, N);

% Add boundary zeros
U_full = zeros(N+2, N+2);
U_full(2:end-1,2:end-1) = U;

%---------------------------------------------------------
% Plot solution
%---------------------------------------------------------
[X,Y] = meshgrid(x,y);

figure
surf(X,Y,U_full')
title('Numerical Solution')
xlabel('x'); ylabel('y');
shading interp

%---------------------------------------------------------
% Compute error
%---------------------------------------------------------
U_exact_vals = u_exact(X,Y);
error = max(max(abs(U_full' - U_exact_vals)));

disp(['Maximum error = ', num2str(error)]);