%=========================================================
% 1D Heat Problem: u_t - u_xx = f on (0,1)x(0,1)
% Comparison of FTCS, BTCS, and Crank-Nicolson methods
%=========================================================
clear
clc
close all

%% Exact solution and source term
u = @(x,t) exp(t).*sin(x);
f = @(x,t) 2.*exp(t).*sin(x);

%% Problem parameters
a = 0;
b = 1;
t0 = 0;
T = 1;

%% Parameters for convergence study
h_values = [1/10, 1/20, 1/40, 1/80 1/160 1/320];  % Spatial step sizes
dt_FTCS = 1e-5;  % Time step for FTCS (needs to be small for stability)
dt_BTCS = 1e-2;  % Time step for BTCS
dt_CN = 1e-2;    % Time step for Crank-Nicolson

%% Initialize results arrays
errors_FTCS = zeros(length(h_values), 1);
errors_BTCS = zeros(length(h_values), 1);
errors_CN = zeros(length(h_values), 1);
order_FTCS = zeros(length(h_values), 1);
order_BTCS = zeros(length(h_values), 1);
order_CN = zeros(length(h_values), 1);

%% Run simulations for different h values
for i = 1:length(h_values)
    h = h_values(i);
    dt = h;
    
    fprintf('\n========================================\n');
    fprintf('Computing for h = %g\n', h);
    fprintf('========================================\n');
    
    %% FTCS Method
    fprintf('Running FTCS method...\n');
    [errors_FTCS(i), ~] = FTCS_method(u, f, a, b, t0, T, h, dt_FTCS);
    
    %% BTCS Method
    fprintf('Running BTCS method...\n');
    [errors_BTCS(i), ~] = BTCS_method(u, f, a, b, t0, T, h, dt^2);
    
    %% Crank-Nicolson Method
    fprintf('Running Crank-Nicolson method...\n');
    [errors_CN(i), ~] = CN_method(u, f, a, b, t0, T, h, dt);
end

%% Calculate computational orders
for i = 2:length(h_values)
    order_FTCS(i) = log(errors_FTCS(i-1)/errors_FTCS(i)) / log(2);
    order_BTCS(i) = log(errors_BTCS(i-1)/errors_BTCS(i)) / log(2);
    order_CN(i) = log(errors_CN(i-1)/errors_CN(i)) / log(2);
end

%% Display results in table format
fprintf('\n\n');
fprintf('================================================================================\n');
fprintf('                  COMPARISON OF FINITE DIFFERENCE METHODS\n');
fprintf('================================================================================\n');
fprintf('%-12s %-20s %-20s %-20s\n', 'h', 'FTCS Error', 'BTCS Error', 'CN Error');
fprintf('--------------------------------------------------------------------------------\n');
for i = 1:length(h_values)
    fprintf('%-12s %-20.6e %-20.6e %-20.6e\n', ...
        num2str(h_values(i)), errors_FTCS(i), errors_BTCS(i), errors_CN(i));
end
fprintf('================================================================================\n\n');

fprintf('================================================================================\n');
fprintf('                  COMPUTATIONAL ORDERS OF CONVERGENCE\n');
fprintf('================================================================================\n');
fprintf('%-12s %-20s %-20s %-20s\n', 'h', 'FTCS Order', 'BTCS Order', 'CN Order');
fprintf('--------------------------------------------------------------------------------\n');
fprintf('%-12s %-20s %-20s %-20s\n', '1/10', '-', '-', '-');
for i = 2:length(h_values)
    fprintf('%-12s %-20.2f %-20.2f %-20.2f\n', ...
        ['1/' num2str(1/h_values(i))], order_FTCS(i), order_BTCS(i), order_CN(i));
end
fprintf('================================================================================\n');

%% Plot comparison of solutions for the finest grid
h_fine = min(h_values);
plot_comparison(u, f, a, b, t0, T, h_fine, dt_FTCS, dt_BTCS, dt_CN);

%% ========================================================================
%% Function definitions
%% ========================================================================

function [error, U_full] = FTCS_method(u, f, a, b, t0, T, h, dt)
    % FTCS method for heat equation
    N = round((b-a)/h);
    M = round((T-t0)/dt);
    x = linspace(a, b, N+1)';
    
    % Initial condition
    U = u(x,0);
    U_new = U;
    
    % Time loop
    s = dt/(h^2);
    for n = 1:M
        tn = n*dt;
        U_new(2:end-1) = U(2:end-1) + s*(U(3:end) - 2*U(2:end-1) + U(1:end-2)) + dt*f(x(2:end-1),tn);
        U_new(1) = u(a,tn);
        U_new(end) = u(b,tn);
        U = U_new;
    end
    
    U_full = U;
    error = norm(U_full - u(x,T), inf);
end

function [error, U_full] = BTCS_method(u, f, a, b, t0, T, h, dt)
    % BTCS method for heat equation
    N = round((b-a)/h);
    M = round((T-t0)/dt);
    s = dt/(h^2);
    x = linspace(a, b, N+1)';
    
    % Initial condition (interior points only)
    U = u(x(2:end-1),0);
    
    % Define matrix A
    main_diag = ones(N-1, 1) * (1+2*s);
    super_diag = ones(N-1, 1) * (-s);
    sub_diag = ones(N-1, 1) * (-s);
    A = spdiags([sub_diag, main_diag, super_diag], [-1, 0, 1], N-1, N-1);
    
    % Time loop
    for n = 1:M
        F = U + dt*f(x(2:end-1), n*dt);
        F(1) = F(1) + s*u(a, n*dt);
        F(end) = F(end) + s*u(b, n*dt);
        U = A\F;
    end
    
    % Add boundary points for full solution
    U_full = zeros(N+1, 1);
    U_full(1) = u(a,T);
    U_full(2:end-1) = U;
    U_full(end) = u(b,T);
    
    error = norm(U_full - u(x,T), inf);
end

function [error, U_full] = CN_method(u, f, a, b, t0, T, h, dt)
    % Crank-Nicolson method for heat equation
    N = round((b-a)/h);
    M = round((T-t0)/dt);
    s = dt/(2*h^2);
    x = linspace(a, b, N+1)';
    
    % Initial condition (interior points only)
    U = u(x(2:end-1),0);
    
    % Define matrices A and B
    main_diag_A = ones(N-1, 1) * (1+2*s);
    super_diag_A = ones(N-1, 1) * (-s);
    sub_diag_A = ones(N-1, 1) * (-s);
    A = spdiags([sub_diag_A, main_diag_A, super_diag_A], [-1, 0, 1], N-1, N-1);
    
    main_diag_B = ones(N-1, 1) * (1-2*s);
    super_diag_B = ones(N-1, 1) * (s);
    sub_diag_B = ones(N-1, 1) * (s);
    B = spdiags([sub_diag_B, main_diag_B, super_diag_B], [-1, 0, 1], N-1, N-1);
    
    % Time loop
    for n = 1:M
        F = B*U + dt*f(x(2:end-1), (n-0.5)*dt);
        F(1) = F(1) + s*u(a, n*dt) + s*u(a, (n-1)*dt);
        F(end) = F(end) + s*u(b, n*dt) + s*u(b, (n-1)*dt);
        U = A\F;
    end
    
    % Add boundary points for full solution
    U_full = zeros(N+1, 1);
    U_full(1) = u(a,T);
    U_full(2:end-1) = U;
    U_full(end) = u(b,T);
    
    error = norm(U_full - u(x,T), inf);
end

function plot_comparison(u, f, a, b, t0, T, h, dt_FTCS, dt_BTCS, dt_CN)
    % Plot comparison of all three methods
    x = linspace(a, b, round((b-a)/h)+1)';
    
    [~, U_FTCS] = FTCS_method(u, f, a, b, t0, T, h, dt_FTCS);
    [~, U_BTCS] = BTCS_method(u, f, a, b, t0, T, h, dt_BTCS);
    [~, U_CN] = CN_method(u, f, a, b, t0, T, h, dt_CN);
    
    % Ensure all vectors have the same length
    fprintf('Vector lengths: x=%d, FTCS=%d, BTCS=%d, CN=%d\n', ...
        length(x), length(U_FTCS), length(U_BTCS), length(U_CN));
    
    figure('Position', [100, 100, 800, 600]);
    plot(x, u(x,T), 'k-', 'LineWidth', 2, 'DisplayName', 'Exact Solution');
    hold on;
    plot(x, U_FTCS, 'ro', 'MarkerSize', 6, 'DisplayName', 'FTCS');
    plot(x, U_BTCS, 'bs', 'MarkerSize', 6, 'DisplayName', 'BTCS');
    plot(x, U_CN, 'gd', 'MarkerSize', 6, 'DisplayName', 'Crank-Nicolson');
    hold off;
    
    xlabel('x', 'FontSize', 12);
    ylabel('u(x,t) at t=1', 'FontSize', 12);
    title(['Comparison of Numerical Methods (h = ' num2str(h) ')'], 'FontSize', 14);
    legend('Location', 'best');
    grid on;
    
    % Zoom in on a region to see differences
    figure('Position', [150, 150, 800, 600]);
    plot(x, u(x,T), 'k-', 'LineWidth', 2, 'DisplayName', 'Exact Solution');
    hold on;
    plot(x, U_FTCS, 'ro', 'MarkerSize', 8, 'DisplayName', 'FTCS');
    plot(x, U_BTCS, 'bs', 'MarkerSize', 8, 'DisplayName', 'BTCS');
    plot(x, U_CN, 'gd', 'MarkerSize', 8, 'DisplayName', 'Crank-Nicolson');
    hold off;
    
    xlim([0.4, 0.6]);
    ylim([1.5, 2.1]);
    xlabel('x', 'FontSize', 12);
    ylabel('u(x,t) at t=1', 'FontSize', 12);
    title('Zoomed View', 'FontSize', 14);
    legend('Location', 'best');
    grid on;
end