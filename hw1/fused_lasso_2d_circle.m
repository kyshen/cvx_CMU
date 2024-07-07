clear,clc

Y = load("circle.csv");
[m,n] = size(Y);

lambda = 1;
A_up    =   eye(m) - diag(ones(m-1,1), -1); A_up(1,1) = 0;
A_down  =   eye(m) - diag(ones(m-1,1), 1);  A_down(end,end) = 0;
A_left  =   eye(n) - diag(ones(n-1,1), 1);  A_left(1,1) = 0;
A_right =   eye(n) - diag(ones(n-1,1), -1); A_right(end,end) = 0;
cvx_begin
    variable X(m,n)
    obj1 = 0.5 * sum(vec(power(Y - X, 2)));
    E = (A_up+A_down)*X + X*(A_left+A_right);
    obj2 = sum(sum(abs(E)));
    minimize(obj1 + lambda*obj2)
cvx_end

subplot 121
imshow(Y)
title("origin")
subplot 122
imshow(X)
