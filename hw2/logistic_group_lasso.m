clear,clc

load Q4c_movies/moviesGroups.mat
load Q4c_movies/moviesTrain.mat
load Q4c_movies/moviesTest.mat

X = trainRatings;
y = trainLabels;
X_test = testRatings;
y_test = testLabels;

[n,p] = size(X);
[n_test,~] = size(X_test);
X = [ones(n,1),X];
X_test = [ones(n_test,1),X_test];

lambda = 5;
t = 1e-4;
steps = 1000;
n_group = length(groupTitles);
groups = ones(1,n_group);
for i = 1:n_group
    groups(i) = length(find(groupLabelsPerRating==i));
end

optimal_value = 336.207;

% PGD
beta = zeros(1+p,1);
error1 = zeros(steps,1);
for k = 1:steps
    beta = prox_h(beta - t * grad_g(beta, X, y), t, lambda, groups);
    error1(k) = obj(beta, X, y, lambda, groups) - optimal_value;
end
sol1 = beta;

% Accelerated PGD
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

% PGD with backtracking line search
beta = zeros(1+p,1);
error3 = zeros(steps,1);
cnt = 1;
f = obj(beta, X, y, lambda, groups);
tk = 1;
for k = 1:400
    while true
        beta1 = prox_h(beta - tk * grad_g(beta, X, y), tk, lambda, groups);
        f1 = obj(beta1, X, y, lambda, groups);
        if f1 > f + grad_g(beta,X,y)'*(beta1-beta)+1/(2*tk)*norm(beta1-beta)
            tk = 0.1*tk;
            error3(cnt) = f - optimal_value;
            cnt = cnt + 1;
        else
            beta = beta1;
            f = f1;
            error3(cnt) = f - optimal_value;
            cnt = cnt + 1;
            break
        end
    end
end
sol3 = beta;

% Predict
a = exp(X_test*sol2);
y_pre = a./(1+a);
y_pre(y_pre>=0.5) = 1;
y_pre(y_pre~=1) = 0;
acc = sum(y_pre==y_test) / n_test;

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
    r = exp(X*beta);
    gradx = -X'*y + X'*(r./(1+r));
end

function value = obj(beta, X, y, lambda, groups)
    value = -y'*X*beta + sum(log(1+exp(X*beta)));
    J = length(groups);
    i_lo = 2;
    for j = 1:J
        w = sqrt(groups(j));
        i_hi = i_lo + groups(j) - 1;
        value = value + lambda * w * norm(beta(i_lo:i_hi), "fro");
        i_lo = i_hi + 1;
    end
end