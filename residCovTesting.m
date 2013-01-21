%% Testing estimation of residual covariance
% AE 2013-01-18


% rng(1)
[gpfa, Y, X, S] = GPFA.toyExample(1e3);
gpfa.M = gpfa.T;
gpfa.means = 'hist';

p = gpfa.p;
q = gpfa.q;
T = gpfa.T;
N = size(Y, 3);


%% fit model
model = GPFA('Tolerance', 1e-6);
model = model.fit(Y, gpfa.p, 'hist');


%% residual covariance
Q = model.residCovByTrial(Y);
imagesc(Q)
colorbar


%% Empirical Sigma = Cov[x, z]
XZ = [reshape(X, [p * T, N]); permute(sum(Y, 2), [1 3 2])];
ndx = [1 : 2 : 40, 2 : 2 : 40, 41 : 56];
Sig = cov(XZ');
% Sig = Sig(ndx, ndx);


%%
Q = T * gpfa.R;
Kb = zeros(p * T);
for i = 1 : p
    ndx = i : p : p * T;
    Kb(ndx, ndx) = toeplitz(gpfa.covFun(0 : T - 1, gpfa.gamma(i)));
end
Ct = repmat(gpfa.C, 1, T);
Sigm = [Kb, Kb * Ct'; Ct * Kb, Ct * Kb * Ct' + Q];


IQ = inv(Q);
ISigm = [Ct' * IQ * Ct + inv(Kb), -Ct' * IQ; -IQ * Ct, IQ];


%% Sigma = Cov[x, z]
Sigma = zeros(p * T + q);
% Q = 10 * cov(randn(20, q));
Q = T * gpfa.R;
Sigma(end - q + 1 : end, end - q + 1 : end) = Q;
for i = 1 : p
    K = toeplitz(gpfa.covFun(0 : T - 1, gpfa.gamma(i)));
    Cxz = kron(sum(K, 1), gpfa.C(:, i));
    ndx = (i - 1) * T + (1 : T);
    Sigma(end - q + 1 : end, ndx) = Cxz;
    Sigma(ndx, end - q + 1 : end) = Cxz';
    Sigma(ndx, ndx) = K;
    Sigma(end - q + 1 : end, end - q + 1 : end) = ...
        Sigma(end - q + 1 : end, end - q + 1 : end) + gpfa.C(:, i) * gpfa.C(:, i)' * sum(K(:));
end

IS = inv(Sigma);

%% Sigma^-1
for i = 1 : p
    Ct = repmat(gpfa.C(:, i), 1, T);
    ndx = (i - 1) * T + (1 : T);
    K = toeplitz(gpfa.covFun(0 : T - 1, gpfa.gamma(i)));
    IS(ndx, ndx) = inv(K) + Ct' / Q * Ct;
    IS(ndx, end - q + 1 : end) = -Ct' / Q;
    IS(end - q + 1 : end, ndx) = -Q \ Ct;
end
IS(end - q + 1 : end, end - q + 1 : end) = inv(Q);
