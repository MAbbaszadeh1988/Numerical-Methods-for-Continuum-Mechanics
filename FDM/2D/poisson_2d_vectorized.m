function [U, x, y] = poisson_2d_vectorized(f, g, Lx, Ly, nx, ny, max_iter, tol)
% Vectorized version for better performance

% Grid setup
dx = Lx / (nx - 1);
dy = Ly / (ny - 1);
x = linspace(0, Lx, nx);
y = linspace(0, Ly, ny);
[X, Y] = meshgrid(x, y);

% Initialize
U = zeros(ny, nx);
U(1,:) = g(X(1,:), Y(1,:));      % Bottom
U(end,:) = g(X(end,:), Y(end,:)); % Top
U(:,1) = g(X(:,1), Y(:,1));       % Left
U(:,end) = g(X(:,end), Y(:,end)); % Right

% Source term at interior points
F = f(X(2:end-1, 2:end-1), Y(2:end-1, 2:end-1));

% Coefficients
alpha = dy^2 / (2*(dx^2 + dy^2));
beta = dx^2 / (2*(dx^2 + dy^2));
gamma = dx^2*dy^2 / (2*(dx^2 + dy^2));

% Jacobi iteration
for iter = 1:max_iter
    U_old = U;
    
    % Vectorized update
    U(2:end-1, 2:end-1) = alpha*(U_old(2:end-1, 3:end) + U_old(2:end-1, 1:end-2)) + ...
                          beta*(U_old(3:end, 2:end-1) + U_old(1:end-2, 2:end-1)) - ...
                          gamma*F;
    
    if norm(U - U_old, 'fro') < tol
        fprintf('Converged after %d iterations\n', iter);
        break;
    end
end

% Plot results
figure;
surf(X, Y, U);
xlabel('x');
ylabel('y');
zlabel('u(x,y)');
title('Solution of Poisson Equation');
colorbar;
end