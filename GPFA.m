classdef GPFA
    % Gaussian Process Factor Analysis with partially observed factors.
    %
    % This is a generalized version of the method described in Yu et al.
    % 2009, J. Neurophys. The implementation is based on and heavily
    % influenced by Byron Yu and John Cunningham's code.
    %
    % Alexander S. Ecker
    % 2013-01-17
    
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
        means       % method of estimating means (zero|hist|reg)
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
            %   SigmaN:     Innovation noise variance for the latent 
            %               Gaussian Process (default: 0.001)
            %   Tolerance:  Stopping criterion used for fitting (default:
            %               0.0001)
            %   Verbose:    Verbose output? (default: false)
            
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
            %   latent factors. Y is assumed to residuals (i.e. the mean
            %   for each bin across trials has been subtracted). Y is a 3d
            %   array of size #neurons x #bins x #trials.
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
                means = 'zero';
            elseif ischar(S) && strcmp(S, 'hist')
                M = T;
                S = [];
                means = 'hist';
            else
                [M, Tc] = size(S);
                means = 'reg';
                assert(T == Tc, 'The number of columns in S and Y must be the same!')
            end
            assert(q > p, 'Number of latent factors must be smaller than number of neurons.')
            
            self.q = q;
            self.T = T;
            self.M = M;
            self.p = p;
            self.means = means;
            
            if nargin < 5
                Yn = reshape(Y, q, T * N);

                switch means
                    case 'zero'
                        D = [];
                        Y0 = Yn;
                    case 'hist'
                        D = mean(Y, 3);
                        Y0 = Yn - repmat(D, 1, N);
                    case 'reg'
                        % initialize stimulus weights using linear regression
                        Sn = repmat(S, 1, N);
                        D = Yn / Sn;
                        Y0 = Yn - D * Sn;
                end
                
                % initialize factor loadings using PCA
                Q = cov(Y0');
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
            self = self.reorderFactors();
            self.runtime = (now() - self.runtime) * 24 * 3600 * 1000; % ms
        end
        
        
        function [EX, VarX, logLike] = estX(self, Y)
            % Estimate latent factors (and log-likelihood).
            %   [EX, VarX, logLike] = self.estX(Y) returns the expected
            %   value (EX) of the latent state X, its variance (VarX), and
            %   the log-likelihood (logLike) of the data Y.
            
            T = self.T; q = self.q; p = self.p; C = self.C; R = self.R;
            N = size(Y, 3);
            
            % catch independent case
            if p == 0
                EX = zeros(0, T, N);
                VarX = [];
                if nargout == 3
                    Y = reshape(Y, q, T * N);
                    val = N * T * (sum(log(diag(R))) + q * log(2 * pi));
                    normY = bsxfun(@rdivide, Y, sqrt(diag(R)));
                    logLike = -0.5 * (val + normY(:)' * normY(:));
                end
                return
            end
            
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
            Y0 = self.subtractMean(Y);
            Y0 = reshape(Y0, q * T, N);
            EX = KbCb * CKCRi * Y0;
            EX = reshape(EX, [p T N]);
            
            % calculate log-likelihood
            if nargout > 2
                Y0 = reshape(Y0, q, T * N);
                val = -T * sum(log(diag(R))) - logdetKb - logdetM - ...
                    q * T * log(2 * pi);
                normY0 = bsxfun(@rdivide, Y0, sqrt(diag(R)));
                CRiY0 = reshape(RiC' * Y0, p * T, N);
                logLike = 0.5 * (N * val - normY0(:)' * normY0(:) + ...
                    sum(sum(CRiY0 .* (VarX * CRiY0))));
            end
        end
        
        
        function Y0 = subtractMean(self, Y)
            % Subtract mean.
            
            switch self.means
                case 'zero'
                    Y0 = Y;
                case 'hist'
                    Y0 = bsxfun(@minus, Y, self.D);
                case 'reg'
                    Y0 = bsxfun(@minus, Y, self.D * self.S);
            end
        end


        function Y = addMean(self, Y0)
            % Add mean.

            switch self.means
                case 'zero'
                    Y = Y0;
                case 'hist'
                    Y = bsxfun(@plus, Y0, self.D);
                case 'reg'
                    Y = bsxfun(@plus, Y0, self.D * self.S);
            end
        end


        function [Yres, X] = resid(self, Y)
            % Compute residuals after accounting for internal factors.
            
            [Ypred, X] = predict(self, Y);
            Yres = Y - Ypred;
        end
        
        
        function [R, X] = residCov(self, Y, byTrial)
            % Residual covariance.
            %   R = model.residCov(Y) returns the residual covariance using
            %   data Y after accounting for latent factors.
            %
            %   R = model.residCov(Y, true) returns the residual covariance
            %   for spike counts summed over the entire trial.
            %
            %   Note: this residuals covariance is computed using the
            %         update rule of the EM algorithm. It is not the same
            %         as computing the covariance of the residuals as in
            %         cov(model.resid(Y)).

            if nargin < 3 || ~byTrial
                [R, X] = self.residCovByBin(Y);
            else
                [R, X] = self.residCovByTrial(Y);
            end
        end


        function [Ypred, X] = predict(self, Y)
            % Prediction of activity based on inference of latent factors.
            
            Ypred = zeros(size(Y));
            N = size(Y, 3);
            X = self.estX(Y);
            for i = 1 : N
                Ypred(:, :, i) = self.C * X(:, :, i);
            end
            Ypred = self.addMean(Ypred);
        end


        function ve = varExpl(self, Y, byTrial)
            % Variance explained by model.
            %   ve = model.varExpl(Y) computes the fraction of variance
            %   explained by the model.
            %
            %   ve = model.varExpl(Y, true) uses spike counts summed over
            %   the entire trial to compute the fraction of variance
            %   explained.

            if nargin < 3 || ~byTrial
                ve = self.varExplByBin(Y);
            else
                ve = self.varExplByTrial(Y);
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
            %
            %   Caution: After applying this transformation, the model
            %   cannot be used for inference on new data.
            
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
            %
            %   Caution: Covariance function is not adjusted properly.
            %   After applying this transformation inference of latent
            %   factors won't be correct any more. This can be fixed but
            %   hasn't been done yet.
            
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
            %
            %   Caution: Covariance function is not adjusted properly.
            %   After applying this transformation inference of latent
            %   factors won't be correct any more. This can be fixed but
            %   hasn't been done yet.
            
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
            
            % catch independent case
            if p == 0
                [~, ~, self.logLike] = self.estX(Y);
                return
            end
            
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
                    switch self.means
                        case 'hist'
                            T1(:, p + t) = sum(y, 2);
                            sx = sum(x, 2);
                            T2(1 : p, p + t) = sx;
                            T2(p + t, 1 : p) = sx';
                            T2(p + t, p + t) = N;
                        case 'reg'
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
                
                Y0 = self.subtractMean(Y);
                Y0 = reshape(Y0, q, T * N);
                self.R = diag(mean(Y0 .^ 2, 2) - ...
                    sum(bsxfun(@times, Y0 * reshape(EX, p, T * N)', self.C), 2) / (T * N));
                
                % optimize gamma
                self.gamma = zeros(p, 1);
                for i = 1 : p
                    ndx = i : p : T * p;
                    EXi = permute(EX(i, :, :), [2 3 1]);
                    EXX = VarX(ndx, ndx) + (EXi * EXi') / N;
                    fun = @(gamma) self.Egamma(gamma, EXX);
                    self.gamma(i) = minimize(self.gamma(i), fun, -25);
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
        

        function self = reorderFactors(self)
            % Re-order factors according to covariance explained.

            C = self.C; p = self.p;
            v = zeros(p, 1);
            for i = 1 : p
                v(i) = mean(mean(C(:, i) * C(:, i)'));
            end
            [~, order] = sort(v, 'descend');
            C = C(:, order);
            C = bsxfun(@times, C, sign(median(C, 1)));   % flip sign?
            self.C = C;
        end


        function [R, X] = residCovByBin(self, Y)
            % Residual covariance for spike counts per bin.

            T = self.T; N = size(Y, 3); p = self.p; q = self.q; C = self.C;
            [X, VarX] = self.estX(Y);
            Y0 = self.subtractMean(Y);
            Y0 = reshape(Y0, q, T * N);
            EXX = 0;
            for t = 1 : T
                x = permute(X(:, t, :), [1 3 2]);
                tt = (1 : p) + p * (t - 1);
                EXX = EXX + N * VarX(tt, tt) + x * x';
            end
            X = reshape(X, p, T * N);
            Y0XC = (Y0 * X') * C';
            R = (Y0 * Y0' - Y0XC - Y0XC' + C * EXX * C') / (T * N);
        end


        function [R, X] = residCovByTrial(self, Y)
            % Residual covariance for spike counts over entire trial.

            T = self.T; N = size(Y, 3); p = self.p; C = self.C;
            [X, VarX] = self.estX(Y);
            X = reshape(X, [p * T, N]);
            Y0 = self.subtractMean(Y);
            Z = permute(sum(Y0, 2), [1 3 2]);
            Ct = repmat(C, 1, T);
            CXZ = Ct * X * Z';
            EXX = N * VarX + X * X';
            R = (Z * Z' - CXZ - CXZ' + Ct * EXX * Ct') / N;
        end


        function ve = varExplByBin(self, Y)
            % Compute variance explained for spike counts per bin..

            Y0 = self.subtractMean(Y);
            V = mean(Y0(:, :) .^ 2, 2);
            R = self.residCovByBin(Y);
            ve = 1 - diag(R) ./ V;
        end


        function ve = varExplByTrial(self, Y)
            % Compute variance explained for spike counts of entire trial.

            Y0 = self.subtractMean(Y);
            V = mean(sum(Y0, 2) .^ 2, 3);
            R = self.residCovByTrial(Y);
            ve = 1 - diag(R) ./ V;
        end

    end
    
    
    methods (Static)
        
        function [model, Y, X, S] = toyExample(N)
            % Create toy example for testing
            %   [model, Y, X, S] = toyExample(N) creates a simple toy
            %   example with N neurons and two latent factors.
            
            if ~nargin
                N = 100;
            end
            T = 20;
            p = 2;
            q = 16;
            gamma = log(1 ./ [4; 1] .^ 2);

            model = GPFA();
            
            K = toeplitz(model.covFun(0 : T - 1, gamma(1)));
            X1 = chol(K)' * randn(T, N);
            K = toeplitz(model.covFun(0 : T - 1, gamma(2)));
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
            
            model = model.collect(C, R, gamma, S, D);
            model.T = T;
            model.p = p;
            model.q = q;
        end

    end
    
end
