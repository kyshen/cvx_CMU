clear,clc
X = h5read("toy.hdf5","/X");
y = h5read("toy.hdf5","/y");

[n,p] = size(X);

C1 = 10;
C2 = 1;

%% primal
H = diag([ ones(p,1); zeros(1+n,1) ]);

v = zeros(size(y));
v(y==1) = C1;
v(y==-1) = C2;
f = [ zeros(p+1,1); v ];

A = zeros(n, p+1+n);
A(:,1:p) = diag(y)*X;
A(:,p+1) = y;
A(:,p+2:end) = eye(n);
A = -A;
b = -1*ones(n,1);

Aeq = [];
beq = [];

lb = [ -inf*ones(p+1,1); zeros(n,1) ];
ub = [];

[x,fval_primal] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
beta = x(1:p);
beta0 = x(p+1);
xi = x(p+2:end);

%% dual
H = diag(y)*X*(diag(y)*X)';
f = -1*ones(n,1);
A = [];
b = [];
Aeq = y';
beq = 0;
lb = zeros(n,1);
ub = v;
[alpha,fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
fval_dual = -fval;

%% visualize
figure(1)
hold on
idx1 = (y==1);
idx0 = (y==-1);
scatter(X(idx1,1), X(idx1,2), 50, 'blue','filled')
scatter(X(idx0,1), X(idx0,2), 50, 'red','filled')
lx = linspace(-4,12,100);
plot(lx, -(beta0+beta(1)*lx)/beta(2), Color="black", LineWidth=2.5)
title(sprintf("C1=%d    C2=%d\n" + ...
              "penalty area = %4.4f",C1,C2,sum(xi)))

figure(2)
hold on
dist = zeros(size(y));
for i = 1:n
    dist(i) = y(i)*(X(i,:)*beta + beta0);
end
xn = 1:n;
plot(xn(idx1), dist(idx1), "b*", LineWidth=2, MarkerSize=8)
plot(xn(idx0), dist(idx0), "r*", LineWidth=2, MarkerSize=8)
plot(xn(idx1), alpha(idx1), "b+", LineWidth=2, MarkerSize=8)
plot(xn(idx0), alpha(idx0), "r+", LineWidth=2, MarkerSize=8)
legend("distance(1)", "distance(-1)", "alpha(1)", "alpha(-1)")
title(sprintf("C1=%d    C2=%d\n" + ...
              "penalty area = %4.4f",C1,C2,sum(xi)))

%% weighted classification error
err_pos = zeros(size(y));
for i = 1:n
    if dist(i) <= 0 % 0 (1) for (strictly) clssify
        err_pos(i) = 1;
    end
end
n_err = sum(err_pos);
weighted_err = sum(err_pos.*v);