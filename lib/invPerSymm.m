function [invM, logdet_M] = invPerSymm(M, q)
% Invert block persymmetric matrix.
%   [invM, logdet_M] = invPerSymm(M, q)
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu
%
% Modified by AE 2012-10-12

% [TODO] 
%   check if we're actually using all knowledge about the structure of M

T = size(M, 1) / q;
mkr = q * ceil(T / 2);
invA11 = inv(M(1 : mkr, 1 : mkr));
invA11 = (invA11 + invA11') / 2;
A12 = M(1 : mkr, mkr + 1 : end);
term = invA11 * A12;
F22 = M(mkr + 1 : end, mkr + 1 : end) - A12' * term;
res12 = -term / F22;
res11 = invA11 - res12 * term';
res11 = (res11 + res11') / 2;
upperHalf = [res11 res12];

% Fill in bottom half of invM by picking elements from res11 and res12
idxHalf = bsxfun(@plus, (1 : q)', (floor(T / 2) - 1 : -1 : 0) * q);
idxFull = bsxfun(@plus, (1 : q)', (T - 1 : -1 : 0) * q);
invM = [upperHalf; upperHalf(idxHalf(:), idxFull(:))];

if nargout == 2
    logdet_M = -logdet(invA11) + logdet(F22);
end
