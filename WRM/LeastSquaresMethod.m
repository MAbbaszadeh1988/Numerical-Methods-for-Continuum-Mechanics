%% Least Square Method for -u'' = M0/EI with u(0)=0, u(1)=0
clear
clc
close all
tic
%% Parameters
M0 = 1;
EI = 1;
L0 = 0;
L = 1;
n = 20;  % Number of basis functions (excluding boundary conditions)

%% Exact solution
u_exact = @(x) -(M0/(2*EI)).*x.*(L-x);

%% Basis functions (that satisfy boundary conditions)
% Using sine functions that automatically satisfy u(0)=u(1)=0
phi = @(x, j) sin(j*pi*x/L);
dxxphi = @(x, j) -(j*pi/L)^2 * sin(j*pi*x/L);

%% Least Squares Method
% We want to find coefficients c_j such that u_N = sum(c_j * phi_j)
% minimizes the residual R = u_N'' + M0/EI

% Compute the matrix A and vector b for the normal equations
A = zeros(n, n);  % A=K
b = zeros(n, 1);

f_rhs = M0/EI;  % Right-hand side constant

for i = 1:n
    for j = 1:n
        % A_ij = ∫ (phi_i'') * (phi_j'') dx
        % trapz, quad, int, integral, ....
        integrand = @(x) dxxphi(x, i) .* dxxphi(x, j);
        A(i, j) = integral(integrand, L0, L, 'ArrayValued', true);
    end
    
    % b_i = -∫ (phi_i'') * f dx
    integrand_b = @(x) dxxphi(x, i) * f_rhs;
    b(i) = integral(integrand_b, L0, L, 'ArrayValued', true);
end

%% Solve for coefficients
c = A \ b;

%% Construct numerical solution
x_plot = linspace(L0, L, 10)';  % Fine grid for plotting
u_numerical = zeros(size(x_plot));
for j = 1:n
    u_numerical = u_numerical + c(j) * phi(x_plot, j);
end

%% Create interpolation function for error calculation
u_numerical_interp = @(xq) interp1(x_plot, u_numerical, xq, 'pchip');

%% Calculate error
u_exact_plot = u_exact(x_plot);
max_error = max(abs(u_numerical - u_exact_plot));
l2_error = sqrt(integral(@(x) (u_numerical_interp(x) - u_exact(x)).^2, L0, L));

fprintf('========================================\n');
fprintf('Least Squares Method Results\n');
fprintf('========================================\n');
fprintf('Number of basis functions: %d\n', n);
fprintf('Maximum error: %e\n', max_error);
fprintf('L2 error: %e\n', l2_error);
fprintf('========================================\n');

%% Plot results
figure('Position', [100, 100, 900, 400]);

% Solution plot
plot(x_plot, u_exact_plot, 'bo', 'LineWidth', 2, 'DisplayName', 'Exact');
hold on;
plot(x_plot, u_numerical, 'r--', 'LineWidth', 2, 'DisplayName', 'Least Squares');
plot(x_plot, u_numerical, 'ro', 'MarkerSize', 2, 'DisplayName', '');
hold off;
xlabel('x', 'FontSize', 12);
ylabel('u(x)', 'FontSize', 12);
title('Solution Comparison', 'FontSize', 14);
legend('Location', 'best');
grid on;
toc