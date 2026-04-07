% example_poisson.m
% Example script for solving Poisson equation

clear; clc; close all;

% Define domain
Lx = 1; Ly = 1;  % Unit square domain
nx = 100; ny = 100;  % Grid points

% Define source term f(x,y)
f = @(x,y) -2*pi^2*sin(pi*x).*sin(pi*y);  % Example: f = -2π²sin(πx)sin(πy)

% Define boundary conditions
g = @(x,y) zeros(size(x));  % Zero Dirichlet BC

% Solve
% [U, x, y] = poisson_2d(f, g, Lx, Ly, nx, ny, 10000, 1e-6);
% [U, x, y] = poisson_2d_vectorized(f, g, Lx, Ly, nx, ny, 10000, 1e-6);
[U, x, y] = poisson_2d_direct_kron_compact(f, g, Lx, Ly, nx, ny);

% Compare with analytical solution (if known)
[X, Y] = meshgrid(x, y);
U_exact = sin(pi*X).*sin(pi*Y);  % Analytical solution for this example

% Calculate error
error = max(max(abs(U - U_exact)));
fprintf('Maximum error: %e\n', error);

% Plot error
figure;
surf(x, y, abs(U - U_exact));
xlabel('x');
ylabel('y');
zlabel('Error');
title('Numerical Error');
colorbar;