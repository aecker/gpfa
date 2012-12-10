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
        S           % basis functions for PSTH
        gamma       % GP timescales
        tau         % GP timescales as SD (unit: bins)
        C           % factor loadings
        D           % stimulus weights
        R           % independent noise variances
        T           % # time points per trial
        M           % # basis functions for PSTH
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
            
            % initialize from struct?
            if nargin && isstruct(varargin{1})
                s = varargin{1};
                for f = fieldnames(s)'
                    self.(f{1}) = s.(f{1});
                end
                return
            end
            
            % parse optional parameters
            p = inputParser; %#ok<*PROP>
            p.KeepUnmatched = true;
            p.addOptional('SigmaN', 1e-3);
            p.addOptional('Tolerance', 1e-4);
            p.addOptional('Verbose', false);
            p.parse(varargin{:});
            self.params = p.Results;
        end
        
        
        function self = fit(self, Y, p, S, C, D, R, gamma)
            % Fit the model
            %   self = fit(self, Y, p) fits the model to data Y using p
            %   latent factors.
            %
            %   self = fit(self, Y, p, S) additionally uses S as basis
            %   functions for predicting the PSTHs.
            %
            %   See GPFA for optional parameters to use for fitting.
            
            self.runtime = now();
            
            % make sure there are no non-spiking cells
            ok = var(Y(1 : end, :), [], 2) > 1e-10;
            assert(all(ok), 'Non-spiking cell found! Please exclude beforehand.')
            
            % determine dimensionality of the problem
            [q, T, N] = size(Y);
            if nargin < 4 || isempty(S)
                M = 0;
                S = [];
            else
                [M, Tc] = size(S);
                assert(T == Tc, 'The number of columns in S and Y must be the same!')
            end
            assert(q > p, 'Number of latent factors must be smaller than number of neurons.')
            
            self.q = q;
            self.T = T;
            self.M = M;
            self.p = p;
            
            if nargin < 5
                Yn = reshape(Y, q, T * N);

                if M > 0
                    % initialize stimulus weights using linear regression
                    Sn = repmat(S, 1, N);
                    D = Yn / Sn;
                    Yres = Yn - D * Sn;
                else
                    D = [];
                    Yres = Yn;
                end
                
                % initialize factor loadings using PCA
                Q = cov(Yres');
                [C, Lambda] = eigs(Q, p);
                
                % initialize private noise as residual variance not accounted
                % for by PCA and stimulus
                R = diag(diag(Q - C * Lambda * C'));
                
                % initialize gammas
                gamma = log(0.01) * ones(p, 1);
            end
            
            self = self.collect(C, R, gamma, S, D);
            
            % run EM
            self = self.EM(Y);
            self.runtime = (now() - self.runtime) * 24 * 3600 * 1000; % ms
        end
        
        
        function [EX, VarX, logLike] = estX(self, Y)
            % Estimate latent factors (and log-likelihood).
            %   [EX, VarX, logLike] = self.estX(Y)
            
            T = self.T; M = self.M; q = self.q; p = self.p;
            S = self.S; C = self.C; D = self.D; R = self.R;
            N = size(Y, 3);
            
            % compute GP covariance and its inverse
            Kb = zeros(T * p, T * p);
            Kbi = zeros(T * p, T * p);
            logdetKb = 0;
            for i = 1 : p
                K = toeplitz(self.covFun(0 : T - 1, self.gamma(i)));
                ndx = i : p : T * p;
                Kb(ndx, ndx) = K;
                [Kbi(ndx, ndx), logdetK] = invToeplitz(K);
                logdetKb = logdetKb + logdetK;
            end
            
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
            if M > 0
                Yres = bsxfun(@minus, Y, D * S);
            else
                Yres = Y;
            end
            Yres = reshape(Yres, q * T, N);
            EX = KbCb * CKCRi * Yres;
            EX = reshape(EX, [p T N]);
            
            % calculate log-likelihood
            if nargout > 2
                Yres = reshape(Yres, q, T * N);
                val = -T * sum(log(diag(R))) - logdetKb - logdetM - ...
                    q * T * log(2 * pi);
                normYres = bsxfun(@rdivide, Yres, sqrt(diag(R)));
                CRiYres = reshape(RiC' * Yres, p * T, N);
                logLike = 0.5 * (N * val - normYres(:)' * normYres(:) + ...
                    sum(sum(CRiYres .* (VarX * CRiYres))));
            end
        end
        
        
        function [Yres, X] = resid(self, Y)
            % Compute residuals after accounting for internal factors.
            
            Yres = zeros(size(Y));
            N = size(Y, 3);
            X = self.estX(Y);
            for i = 1 : N
                Yres(:, :, i) = Y(:, :, i) - self.C * X(:, :, i);
            end
        end
        
        function k = covFun(self, t, gamma)
            % Gaussian process covariance function
            
            sn = self.params.SigmaN;
            sf = 1 - sn;
            k = sf * exp(-0.5 * exp(gamma) * t .^ 2) + sn * (t == 0);
        end
        
        
        function [self, X] = ortho(self, X)
            % Orthogonalize factor loadings
            
            [self.C, S, V] = svd(self.C, 'econ');
            if nargout > 1
                if size(X, 1) == self.q  % Y passed -> estimte X
                    X = self.estX(X);
                end
                N = size(Y, 3);
                X = reshape(X, self.p, self.T * N);
                X = S * V' * X;
                X = reshape(X, [self.p self.T N]);
            end
        end
        
        
        function [self, X] = normLoadings(self, X)
            % Normalize factor loadings
            
            n = sqrt(sum(self.C .^ 2, 1));
            self.C = bsxfun(@rdivide, self.C, n);
            if nargout > 1
                if size(X, 1) == self.q  % Y passed -> estimte X
                    X = self.estX(X);
                end
                for i = 1 : self.p
                    X(i, :) = X(i, :) * n(i);
                end
            end
        end
        
        
        function [self, X] = normFactors(self, X)
            % Normalize factors to unit variance
            
            if size(X, 1) == self.q  % Y passed -> estimte X
                X = self.estX(X);
            end
            for i = 1 : self.p
                sd = std(X(i, :));
                self.C(:, i) = self.C(:, i) * sd;
                if nargout > 1
                    X(i, :) = X(i, :) / sd;
                end
            end
        end
        
        
        function s = struct(self)
            % Convert to struct.
            
            state = warning('off', 'MATLAB:structOnObject');
            s = builtin('struct', self);
            warning(state)
        end
        
    end
    
    
    methods (Access = protected)
        
        function self = collect(self, C, R, gamma, S, D)
            self.C = C;
            self.R = R;
            self.gamma = gamma;
            if nargin > 4
                self.S = S;
                self.D = D;
            end    
        end
        
        function [E, dEdgamma] = Egamma(self, gamma, EXX)
            % EXX is the average (over N) of the second moments of X
            
            sigmaf = 1 - self.params.SigmaN;
            t = 0 : self.T - 1;
            [Ki, logdetK] = invToeplitz(self.covFun(t, gamma));
            ttsq = bsxfun(@minus, t, t') .^ 2;
            dKdgamma = -0.5 * sigmaf * exp(gamma) * ttsq .* exp(-0.5 * exp(gamma) * ttsq);
            dEdK = 0.5 * (Ki - Ki * EXX * Ki);
            dEdgamma = dEdK(:)' * dKdgamma(:);
            E = 0.5 * (logdetK + EXX(:)' * Ki(:));
        end

    end
    
    
    methods (Access = private)
        
        function self = EM(self, Y)
            % Run EM.
            %   self = self.EM(Y) runs the EM iteration until convergence.
            
            S = self.S; p = self.p; q = self.q; T = self.T; M = self.M;
            N = size(Y, 3);
            Sn = repmat(S, [1 1 N]);
            
            iter = 0;
            logLikeBase = NaN;
            while iter <= 2 || (self.logLike(end) - self.logLike(end - 1)) / (self.logLike(end - 1) - logLikeBase) > self.params.Tolerance
                
                iter = iter + 1;
                
                % E step
                [EX, VarX, self.logLike(end + 1)] = estX(self, Y);
                
                % Perform M step
                T1 = zeros(q, p + M);
                T2 = zeros(p + M);
                for t = 1 : T
                    x = permute(EX(:, t, :), [1 3 2]);
                    y = permute(Y(:, t, :), [1 3 2]);
                    T1(:, 1 : p) = T1(:, 1 : p) + y * x';
                    tt = (1 : p) + p * (t - 1);
                    T2(1 : p, 1 : p) = T2(1 : p, 1 : p) + N * VarX(tt, tt) + x * x';
                    if M > 0
                        s = permute(Sn(:, t, :), [1 3 2]);
                        sx = x * s';
                        T1(:, p + (1 : M)) = T1(:, p + (1 : M)) + y * s';
                        T2(1 : p, p + (1 : M)) = T2(1 : p, p + (1 : M)) + sx;
                        T2(p + (1 : M), 1 : p) = T2(p + (1 : M), 1 : p) + sx';
                        T2(p + (1 : M), p + (1 : M)) = T2(p + (1 : M), p + (1 : M)) + s * s';
                    end
                end
                CD = T1 / T2;
                self.C = CD(:, 1 : p);
                self.D = CD(:, p + (1 : M));
                
                if M > 0
                    Yres = bsxfun(@minus, Y, self.D * S);
                else
                    Yres = Y;
                end
                Yres = reshape(Yres, q, T * N);
                self.R = diag(mean(Yres .^ 2, 2) - ...
                    sum(bsxfun(@times, Yres * reshape(EX, p, T * N)', self.C), 2) / (T * N));
                
                % optimize gamma
                self.gamma = zeros(p, 1);
                for i = 1 : p
                    ndx = i : p : T * p;
                    EXi = permute(EX(i, :, :), [2 3 1]);
                    EXX = VarX(ndx, ndx) + (EXi * EXi') / N;
                    fun = @(gamma) self.Egamma(gamma, EXX);
                    self.gamma(i) = minimize(self.gamma(i), fun, -10);
                end
                self.tau = exp(-self.gamma / 2);

                if iter == 2
                    logLikeBase = self.logLike(end);
                end
                
                if self.params.Verbose
                    subplot(211)
                    plot(self.logLike(2 : end), '.-k')
                    subplot(212), hold all
                    plot(self.C(:, 1), 'k')
                    drawnow
                end
            end
        end
        
    end
    
    
    methods (Static)
        
        function [gpfa, Y, X, S] = toyExample()
            % Create toy example for testing
            
            N = 100;
            T = 20;
            p = 2;
            q = 8;
            gamma = log(1 ./ [4; 1] .^ 2);

            gpfa = GPFA();
            
            K = toeplitz(gpfa.covFun(0 : T - 1, gamma(1)));
            X1 = chol(K)' * randn(T, N);
            K = toeplitz(gpfa.covFun(0 : T - 1, gamma(2)));
            X2 = chol(K)' * randn(T, N);
            X = [X1(:), X2(:)]';
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi)]' / sqrt(q / 2);
            D = rand(q, T);
            S = eye(T);
            Sn = repmat(S, 1, N);
            R = 0.02 * eye(q);
            Y = chol(R)' * randn(q, T * N) + C * X + D * Sn;
            Y = reshape(Y, [q T N]);
            
            gpfa = gpfa.collect(C, R, gamma, S, D);
            gpfa.T = T;
            gpfa.p = p;
            gpfa.q = q;
        end
        
        
        function [gpfa, Y, X, S] = toyExampleOri(noise)
            % Toy example with up/down states being orientation-domain
            % specific.
            
            if ~nargin, noise = 'gauss'; end
        
            N = 200;    % trials
            T = 20;     % bins (e.g. 50 ms -> 1 sec trials)
            p = 3;
            q = 16;

            [b, a] = butter(5, 0.05);
            theta = cumsum(randn(1, N * T));
            theta = filtfilt(b, a, theta);
            ampl = randn(1, N * T);
            ampl = filtfilt(b, a, ampl);
            ampl = 2 ./ (1 + exp(3 * ampl)) - 1;
            X = [cos(theta); sin(theta); ampl];
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi); ones(1, q)]';
            C = bsxfun(@rdivide, C, sqrt(sum(C .^ 2, 1)));
            
            t = sin((linspace(0, 1, T) .^ (1 / 7)) * 3);
            D = exp(cos(phi))' * t;
            
            R = diag(mean(D, 2));
            
            S = eye(T);
            Sn = repmat(S, 1, N);
            
            switch noise
                case 'gauss'
                    Y = chol(R)' * randn(q, T * N) + C * X + D * Sn;
                case 'poisson'
                    Y = poissrnd(max(0, C * X + D * Sn));
            end
            Y = reshape(Y, [q T N]);
            
            gpfa = GPFA();
            gpfa = gpfa.collect(C, R, [], S, D);
            gpfa.T = T;
            gpfa.p = p;
            gpfa.q = q;
        end
    end
    
end
