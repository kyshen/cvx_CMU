clear,clc

c = [30 25 20 50 25 50 25 30]';
E = length(c);
cvx_begin
    variable b(E)
    variable x(6)
    minimize(c'*b)
    subject to
        x(1) - x(6) >= 1;
        b >= 0;
        b(1) >= x(1) - x(2);
        b(2) >= x(1) - x(4);
        b(3) >= x(4) - x(2);
        b(4) >= x(2) - x(3);
        b(5) >= x(4) - x(3);
        b(6) >= x(4) - x(5);
        b(7) >= x(3) - x(6);
        b(8) >= x(5) - x(6);
cvx_end