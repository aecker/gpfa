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
        tau         % GP timescales
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
            p.addOptional('Tolerance', 0.0005);
            p.addOptional('Verbose', false);
            p.parse(varargin{:});
            self.params = p.Results;

            % covariance function
            sn = self.params.SigmaN;
            sf = 1 - sn;
            self.k = @(t, tau) sf * exp(-1/2 * t .^ 2 / tau ^ 2) + sn * (t == 0);
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
            self.Y = Y;
            self.p = p;
            
            % ensure deterministic behavior
            rng(self.params.Seed);
            
            if nargin < 4
                
                % average firing rates
                Y = reshape(Y, q, T * N);
                self.D = mean(Y, 2);
                
                % initialize factor loadings using PCA
                resid = bsxfun(@minus, Y, self.D);
                Q = cov(resid');
                [self.C, Lambda] = eigs(Q, p);
                
                % initialize private noise as residual variance not accounted
                % for by PCA and stimulus
                self.R = diag(diag(Q - self.C * Lambda * self.C'));
                
                % initialize gammas
                self.gamma = log(0.5 / 100) * ones(p, 1);
                %self.tau = exp(randn(p, 1));
            else
                self.C = C;
                self.D = D;
                self.R = R;
                self.gamma = gamma;
            end
            
            % run EM
            self = self.EM();
        end
    end
    
    
    methods (Access = protected)
        
        function [Y, C, D, R, tau, X] = expand(self)
            Y = self.Y;
            C = self.C;
            D = self.D;
            R = self.R;
            tau = self.tau;
            X = self.X;
        end
        
        
        function self = collect(self, Y, C, D, R, tau, X)
            self.Y = Y;
            self.C = C;
            self.D = D;
            self.R = R;
            self.tau = tau;
            self.X = X;
        end
        
        function [E, dEdtau] = Etau(self, tau, EXX)
            % EXX is the sum (over N) of the second moments of X
            
            sigmaf = 1 - self.params.SigmaN;
            N = self.N;
            t = 0 : self.T - 1;
            [Ki, logdetK] = invToeplitz(self.k(t, tau));
            ttsq = bsxfun(@minus, t, t') .^ 2;
            dKdtau = sigmaf ^ 2 * ttsq / tau ^ 3 .* exp(-0.5 * ttsq / tau ^ 2);
            dEdK = 0.5 * (N * Ki - Ki * EXX * Ki);
            dEdtau = dEdK(:)' * dKdtau(:);
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
            [Y, C, D, R, tau] = self.expand();
            
            % pre-compute GP covariance and inverse
            p = self.p;
            q = self.q;
            T = self.T;
            N = self.N;
            S = eye(T);
            [Kb, Kbi] = self.makeKb();
            Yn = reshape(Y, [q T N]);
            
            iter = 0;
            logLikeBase = NaN;
            while iter < maxIter && (iter < 2 || (self.logLike(end) - self.logLike(end - 1)) / (self.logLike(end - 1) - logLikeBase) > self.params.Tolerance)
                
                iter = iter + 1;

                % Perform E step
                RiC = bsxfun(@rdivide, C, diag(R));
                CRiC = C' * RiC;
                VarX = invPerSymm(Kbi + kron(eye(T), CRiC), p);
                RbiCb = kron(eye(T), RiC); % [TODO] can kron be optimized?
                Rbi = kron(eye(T), diag(1 ./ diag(R)));
                Cb = kron(eye(T), C);
                KbCb = Kb * Cb'; % [TODO] optimize: K is block-diagonal
                % KbCb = kron(K, C'); % if all Ks/taus are equal
                CKCRi = Rbi - RbiCb * VarX * RbiCb';
                YDS = bsxfun(@minus, Y, D * S);
                YDS = reshape(YDS, q * T, N);
                EX = KbCb * CKCRi * YDS;
                EX = reshape(EX, [p T N]);

                % calculate log-likelihood
                self.logLike(end + 1) = ...
                    - N * sum(log(diag(chol(CKCRi)))) ...
                    - sum(sum(YDS .* (CKCRi * YDS))) / 2;
                
                % Perform M step
                T1 = zeros(q, p + T);
                T2 = zeros(p + T);
                for t = 1 : T
                    tt = (1 : p) + p * (t - 1);
                    for n = 1 : N
                        x = EX(:, t, n);
                        s = S(:, t);
                        % [TODO] make more efficient: s mostly zero
                        T1 = T1 + Yn(:, t, n) * [x', s'];
                        T2 = T2 + [VarX(tt, tt) + x * x', x * s'; s * x', s * s'];
                    end
                end
                CD = T1 / T2;
                C = CD(:, 1 : p);
                D = CD(:, p + (1 : T));
                
                YDS = reshape(YDS, q, T * N);
                R = diag(mean(YDS .^ 2, 2) - ...
                    sum(bsxfun(@times, YDS * reshape(EX, p, T * N)', C), 2) / (T * N));
                
                % maximize tau
                for i = 1 : p
                    ndx = i : p : T * p;
                    EXi = permute(EX(i, :, :), [2 3 1]);
                    EXX = N * VarX(ndx, ndx) + (EXi * EXi');
                    fun = @(tau) self.Etau(tau, EXX);
                    tau(i) = minimize(tau(i), fun, 50);
                end

                if iter == 1
                    logLikeBase = self.logLike(end);
                end
                
                if self.params.Verbose
                    subplot(211)
                    plot(self.logLike, '.-k')
                    subplot(212), hold all
                    plot(C(:, 1), 'k')
                end
            end
            
            % orthogonalize
            [C, S, V] = svd(C, 'econ');
            EX = reshape(EX, p, T * N);
            EX = S * V' * EX;
            EX = reshape(EX, [p T N]);
            
            self = self.collect(Y, C, D, R, tau, EX);
        end
        
        
        function [Kb, Kbi] = makeKb(self)
            
            T = self.T;
            p = self.p;
            Kb = zeros(T * p, T * p);
            Kbi = zeros(T * p, T * p);
            for i = 1 : p
                K = toeplitz(self.k(0 : T - 1, self.tau(i)));
                ndx = i : p : T * p;
                Kb(ndx, ndx) = K;
                Kbi(ndx, ndx) = invToeplitz(K);
            end
        end
        
    end
    
    
    methods (Static)
        
        function [gpfa, Y] = toyExample()
            % Create toy example for testing
            
            rng(1);
            N = 100;
            T = 20;
            p = 2;
            q = 8;
            tau = [4; 1];

            gpfa = GPFA('SigmaN', 0.1);
            
            K = toeplitz(gpfa.k(0 : T - 1, tau(1)));
            X1 = chol(K)' * randn(T, N);
            K = toeplitz(gpfa.k(0 : T - 1, tau(2)));
            X2 = chol(K)' * randn(T, N);
            X = [X1(:), X2(:)]';
            X = X1(:)';
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi)]' / sqrt(q / 2);
            C = cos(phi)' / sqrt(q / 2);
            D = rand(q, T);
            S = repmat(eye(T), 1, N);
            R = 0.02 * eye(q);
            Y = chol(R)' * randn(q, T * N) + C * X + D * S;
            Y = reshape(Y, [q T N]);
            
            gpfa = gpfa.collect(Y, C, D, R, tau, X);
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

