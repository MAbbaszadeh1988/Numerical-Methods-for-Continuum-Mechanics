%=========================================================
% 1D Heat Problem:  u_t - u_xx = f  on (0,1)x(0,1)
% FTCS finite difference method
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
h = 1/10;
N = (b-a)/h;
t0 = 0;
T = 1;
dt = 1e-4;
M = (T-t0)/dt;
x = (a:h:b)';
t = (t0:dt:T)';
%% Initial condition
U = u(x,0);
U_new = U;  % Store new time step
%% Time loop
s = dt/(h^2);
for n=1:M
    tn = n*dt
    U_new(2:end-1) = U(2:N) + s*(U(3:end) - 2*U(2:end-1) + U(1:end-2)) + dt*f(x(2:end-1),tn);
    U_new(1) = 0;
    U_new(end) = exp(tn)*sin(1);
    U = U_new;  % Update solution
end
norm(U-u(x,T),inf)
plot(x,U,'o',x,u(x,T),'.-','LineWidth',2)
xlabel('x')
ylabel('Exact and Approximate Solution')