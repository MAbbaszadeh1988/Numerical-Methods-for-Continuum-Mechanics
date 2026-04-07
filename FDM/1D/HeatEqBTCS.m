%=========================================================
% 1D Heat Problem:  u_t - u_xx = f  on (0,1)x(0,1)
% BTCS finite difference method
%=========================================================
clear
clc
close all
%% Exact solution and source term
u = @(x,t) exp(t).*sin(x);
f = @(x,t) 2.*exp(t).*sin(x);
%%
a = 0;
b = 1;
h = 1/80;
N = (b-a)/h;
t0 = 0;
T = 1;
dt = 1e-2;
M = (T-t0)/dt;
s = dt/(h^2);
x = (a:h:b)';
t = (t0:dt:T)';
%% Initial condition
U = u(x(2:end-1),0);
%% Define diagonals as columns in a matrix
main_diag = ones(N-1, 1) * (1+2*s);         
super_diag = ones(N-1, 1) * (-s);   
sub_diag = ones(N-1, 1) * (-s);    
A = spdiags([sub_diag, main_diag, super_diag], [-1, 0, 1], N-1, N-1);
%% Time loop
for n=1:M
    n*dt
    F = U + dt*f(x(2:end-1),n*dt);
    F(1)   = F(1)   + s*u(a,n*dt);
    F(end) = F(end) + s*u(b,n*dt);
    U = A\F;
end
norm(U-u(x(2:end-1),T),inf)
plot(x,[u(a,T);U;u(b,T)],'o',x,u(x,T),'.-','LineWidth',2)
xlabel('x')
ylabel('Exact and Approximate Solution')