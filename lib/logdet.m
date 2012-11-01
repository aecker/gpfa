function y = logdet(A)
% log(det(A)) where A is positive-definite.
% This is faster and more stable than using log(det(A)).

U = chol(A);
y = 2 * sum(log(diag(U)));
