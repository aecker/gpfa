classdef GPFA
    % Gaussian Process Factor Analysis with partially observed factors.
    %
    % This is a generalized version of the method described in Yu et al.
    % 2009, J. Neurophys. The implementation is based on and heavily
    % influenced by Byron Yu and John Cunningham's code.
    %
    % Alexander S. Ecker
    % 2012-10-12
    
    properties
        params      % parameters for fitting
        Y           % spike count data
        k           % GP covariance function
        gamma       % GP timescales
        tau         % GP timescales as SD (unit: bins)
        C           % factor loadings
        D           % stimulus weights
        R           % independent noise variances
        X           % latent factors (GP)
        T           % # time points per trial
        N           % # trials
        p           % # unobserved factors
        q           % # neurons
        logLike     % log-likelihood curve during fitting
    end
    
    properties (Access = private)
        runtime     % runtime of fitting process
    end
    
    methods

        function self = GPFA(varargin)
            % GPFA constructor
            %   gpfa = GPFA('param1', value1, 'param2', value2, ...)
            %   constructs a GPFA object with the following optional
            %   parameters:
            %
            %   TODO
            
            % parse optional parameters
            p = inputParser;
            p.KeepUnmatched = true;
            p.addOptional('SigmaN', 1e-3);
            p.addOptional('Seed', 1);
            p.addOptional('Tolerance', 1e-4);
            p.addOptional('Verbose', false);
            p.parse(varargin{:});
            self.params = p.Results;

            % covariance function
            sn = self.params.SigmaN;
            sf = 1 - sn;
            self.k = @(t, gamma) sf * exp(-0.5 * exp(gamma) * t .^ 2) + sn * (t == 0);
        end
        
        
        function self = fit(self, Y, p, C, D, R, gamma)
            % Fit the model
            %   self = fit(self, Y, p) fits the model to data Y using p
            %   latent factors.
            %
            %   See GPFA for optional parameters to use for fitting.
            
            % determine dimensionality of the problem
            [q, T, N] = size(Y);
            
            self.q = q;
            self.T = T;
            self.N = N;
            self.p = p;
            
            % ensure deterministic behavior
            rng(self.params.Seed);
            
            if nargin < 4
                
                self.Y = Y;
                
                % initialize stimulus weights using linear regression
                Y = reshape(Y, q, T * N);
                S = repmat(eye(T), 1, N);
                self.D = Y / S;
                
                % initialize factor loadings using PCA
                resid = Y - self.D * S;
                Q = cov(resid');
                [self.C, Lambda] = eigs(Q, p);
                
                % initialize private noise as residual variance not accounted
                % for by PCA and stimulus
                self.R = diag(diag(Q - self.C * Lambda * self.C'));
                
                % initialize gammas
                self.gamma = log(0.01) * ones(p, 1);
            else
                self = self.collect(Y, C, D, R, gamma, []);
            end
            
            % run EM
            self = self.EM();
        end
        
        
        function self = ortho(self)
            % Orthogonalize factor loadings
            
            [self.C, S, V] = svd(self.C, 'econ');
            X = reshape(self.X, self.p, self.T * self.N);
            X = S * V' * X;
            self.X = reshape(X, [self.p self.T self.N]);
        end
        
        
        function self = normLoadings(self)
            % Normalize factor loadings
            
            for i = 1 : self.p
                n = norm(self.C(:, i));
                self.C(:, i) = self.C(:, i) / n;
                self.X(i, :) = self.X(i, :) * n;
            end
        end
        
        
        function self = normFactors(self)
            % Normalize factors to unit variance
            
            for i = 1 : self.p
                sd = std(self.X(i, :));
                self.X(i, :) = self.X(i, :) / sd;
                self.C(:, i) = self.C(:, i) * sd;
            end
        end
        
    end
    
    
    methods (Access = protected)
        
        function [Y, C, D, R, gamma, X] = expand(self)
            Y = self.Y;
            C = self.C;
            D = self.D;
            R = self.R;
            gamma = self.gamma;
            X = self.X;
        end
        
        
        function self = collect(self, Y, C, D, R, gamma, X)
            self.Y = Y;
            self.C = C;
            self.D = D;
            self.R = R;
            self.gamma = gamma;
            self.X = X;
        end
        
        function [E, dEdgamma] = Egamma(self, gamma, EXX)
            % EXX is the sum (over N) of the second moments of X
            
            sigmaf = 1 - self.params.SigmaN;
            N = self.N;
            t = 0 : self.T - 1;
            [Ki, logdetK] = invToeplitz(self.k(t, gamma));
            ttsq = bsxfun(@minus, t, t') .^ 2;
            dKdgamma = -0.5 * sigmaf * exp(gamma) * ttsq .* exp(-0.5 * exp(gamma) * ttsq);
            dEdK = 0.5 * (N * Ki - Ki * EXX * Ki);
            dEdgamma = dEdK(:)' * dKdgamma(:);
            E = 0.5 * (N * logdetK + EXX(:)' * Ki(:));
%             E = 0.5 * (EXX(:)' * Ki(:));
        end

    end
    
    
    methods (Access = private)
        
        function self = EM(self, maxIter)
            % Run EM.
            %   self = EM(self) runs the EM iteration until convergence.
            %
            %   self = EM(self, maxIter) runs the EM iteration until
            %   convergence but at most maxIter iterations.
            
            if nargin < 2, maxIter = Inf; end
            [Y, C, D, R, gamma] = self.expand();
            
            % pre-compute GP covariance and inverse
            p = self.p;
            q = self.q;
            T = self.T;
            N = self.N;
            S = eye(T);
            Yn = reshape(Y, [q T N]);
            
            iter = 0;
            logLikeBase = NaN;
            while iter < maxIter && (iter < 2 || (self.logLike(end) - self.logLike(end - 1)) / (self.logLike(end - 1) - logLikeBase) > self.params.Tolerance)
                
                iter = iter + 1;
                
                [Kb, Kbi, logdetKb] = self.makeKb(gamma);
            
                % Perform E step
                RiC = bsxfun(@rdivide, C, diag(R));
                CRiC = C' * RiC;
                [VarX, logdetM] = invPerSymm(Kbi + kron(eye(T), CRiC), p);
                RbiCb = kron(eye(T), RiC); % [TODO] can kron be optimized?
                Rbi = kron(eye(T), diag(1 ./ diag(R)));
                Cb = kron(eye(T), C);
                KbCb = Kb * Cb'; % [TODO] optimize: K is block-diagonal
                % KbCb = kron(K, C'); % if all Ks/taus are equal
                CKCRi = Rbi - RbiCb * VarX * RbiCb';
                YDS = bsxfun(@minus, Y, D);
                YDS = reshape(YDS, q * T, N);
                EX = KbCb * CKCRi * YDS;
                EX = reshape(EX, [p T N]);

                % calculate log-likelihood 
                % !!!! [TODO: need to adapt to stim terms] !!!
                YDS = reshape(YDS, q, T * N);
                val = -T * sum(log(diag(R))) - logdetKb - logdetM - ...
                    q * T * log(2 * pi);
                normYDS = bsxfun(@rdivide, YDS, sqrt(diag(R)));
                CRiYDS = reshape(RiC' * YDS, p * T, []);
                self.logLike(end + 1) = 0.5 * (N * val - ...
                    normYDS(:)' * normYDS(:) + sum(sum(CRiYDS .* (VarX * CRiYDS))));
                
                % Perform M step
                T1 = zeros(q, p + T);
                T2 = zeros(p + T);
                for t = 1 : T
                    x = permute(EX(:, t, :), [1 3 2]);
                    y = permute(Yn(:, t, :), [1 3 2]);
                    T1(:, 1 : p) = T1(:, 1 : p) + y * x';
                    T1(:, p + t) = sum(y, 2);
                    tt = (1 : p) + p * (t - 1);
                    sx = sum(x, 2);
                    T2(1 : p, 1 : p) = T2(1 : p, 1 : p) + N * VarX(tt, tt) + x * x';
                    T2(1 : p, p + t) = sx;
                    T2(p + t, 1 : p) = sx';
                    T2(p + t, p + t) = N;
                end
                CD = T1 / T2;
                C = CD(:, 1 : p);
                D = CD(:, p + (1 : T));
                
                R = diag(mean(YDS .^ 2, 2) - ...
                    sum(bsxfun(@times, YDS * reshape(EX, p, T * N)', C), 2) / (T * N));
                
                % maximize gamma
                for i = 1 : p
                    ndx = i : p : T * p;
                    EXi = permute(EX(i, :, :), [2 3 1]);
                    EXX = N * VarX(ndx, ndx) + (EXi * EXi');
                    fun = @(gamma) self.Egamma(gamma, EXX);
                    gamma(i) = minimize(gamma(i), fun, -5);
                end

                if iter == 1
                    logLikeBase = self.logLike(end);
                end
                
                if self.params.Verbose
                    subplot(211)
                    plot(self.logLike, '.-k')
                    subplot(212), hold all
                    plot(C(:, 1), 'k')
                    drawnow
                end
            end
            
            self = self.collect(Y, C, D, R, gamma, EX);
            self.tau = exp(-gamma / 2);
        end
        
        
        function [Kb, Kbi, logdetKb] = makeKb(self, gamma)
            
            T = self.T;
            p = self.p;
            Kb = zeros(T * p, T * p);
            Kbi = zeros(T * p, T * p);
            logdetKb = 0;
            for i = 1 : p
                K = toeplitz(self.k(0 : T - 1, gamma(i)));
                ndx = i : p : T * p;
                Kb(ndx, ndx) = K;
                [Kbi(ndx, ndx), logdetK] = invToeplitz(K);
                logdetKb = logdetKb + logdetK;
            end
        end
        
    end
    
    
    methods (Static)
        
        function [gpfa, Y] = toyExample()
            % Create toy example for testing
            
            N = 100;
            T = 20;
            p = 2;
            q = 8;
            gamma = log(1 ./ [4; 1] .^ 2);

            gpfa = GPFA();
            
            K = toeplitz(gpfa.k(0 : T - 1, gamma(1)));
            X1 = chol(K)' * randn(T, N);
            K = toeplitz(gpfa.k(0 : T - 1, gamma(2)));
            X2 = chol(K)' * randn(T, N);
            X = [X1(:), X2(:)]';
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi)]' / sqrt(q / 2);
            D = rand(q, T);
            S = repmat(eye(T), 1, N);
            R = 0.02 * eye(q);
            Y = chol(R)' * randn(q, T * N) + C * X + D * S;
            Y = reshape(Y, [q T N]);
            
            gpfa = gpfa.collect(Y, C, D, R, gamma, X);
            gpfa.T = T;
            gpfa.N = N;
            gpfa.p = p;
            gpfa.q = q;
        end
        
        
        function [gpfa, Y] = toyExampleOri()
            % Toy example with up/down states being orientation-domain
            % specific.
        
            N = 100;    % trials
            T = 20;     % bins (e.g. 50 ms -> 1 sec trials)
            p = 3;
            q = 16;

            [b, a] = butter(5, 0.05);
            theta = cumsum(randn(1, N * T));
            theta = filtfilt(b, a, theta);
            ampl = randn(1, N * T);
            ampl = filtfilt(b, a, ampl);
            ampl = 2 ./ (1 + exp(3 * ampl));
            X = [cos(theta); sin(theta); ampl];
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi); ones(1, q)]';
            C = bsxfun(@rdivide, C, sqrt(sum(C .^ 2, 1)));
            
            t = sin((linspace(0, 1, T) .^ (1 / 7)) * 3);
            D = exp(cos(phi))' * t;
            
            R = diag(mean(D, 2));
            
            S = repmat(eye(T), 1, N);
            Y = chol(R)' * randn(q, T * N) + C * X + D * S;
            Y = reshape(Y, [q T N]);
            
            gpfa = GPFA();
            gpfa = gpfa.collect(Y, C, D, R, [], X);
            gpfa.T = T;
            gpfa.N = N;
            gpfa.p = p;
            gpfa.q = q;
        end
    end
    
end


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

end

