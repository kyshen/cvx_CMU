clear,clc

load birthwt.mat
X = table2array(X);
y = table2array(y);
[n,p] = size(X);
X = [ones(n,1),X];

lambda = 4;
t = 0.002;
steps = 1000;
groups = [3,3,2,1,2,1,1,3];

optimal_value = 84.6952;

beta = zeros(1+p,1);
error1 = zeros(steps,1);
for k = 1:steps
    beta = prox_h(beta - t * grad_g(beta, X, y), t, lambda, groups);
    error1(k) = obj(beta, X, y, lambda, groups) - optimal_value;
end
sol1 = beta;

beta = zeros(1+p,1);
error2 = zeros(steps,1);
B = {beta,beta};
for k = 1:steps
    v = B{1} + (k-2)/(k+1)*(B{1}-B{2});
    beta = prox_h(v - t * grad_g(beta, X, y), t, lambda, groups);
    error2(k) = obj(beta, X, y, lambda, groups) - optimal_value;
    B{2} = B{1};
    B{1} = beta;
end
sol2 = beta;

plot(log(error1))
hold on
plot(log(error2))


function z = prox_h(x, t, lambda, groups)
    if length(x) ~= sum(groups) + 1
        error("wrong group!")
    end
    z = zeros(size(x));
    z(1) = x(1);
    J = length(groups);
    i_lo = 2;
    for j = 1:J
        w = sqrt(groups(j));
        i_hi = i_lo + groups(j) - 1;
        a = 1-t*lambda*w/norm(x(i_lo:i_hi), "fro");
        z(i_lo:i_hi) = max(0,a)*x(i_lo:i_hi);
        i_lo = i_hi + 1;
    end
end

function gradx = grad_g(beta, X, y)
    gradx = -2*X'*(y-X*beta);
end

function value = obj(beta, X, y, lambda, groups)
    value = norm(y - X*beta)^2;
    J = length(groups);
    i_lo = 2;
    for j = 1:J
        w = sqrt(groups(j));
        i_hi = i_lo + groups(j) - 1;
        value = value + lambda * w * norm(beta(i_lo:i_hi), "fro");
        i_lo = i_hi + 1;
    end
end

