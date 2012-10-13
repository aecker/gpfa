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
        C           % factor loadings
        D           % stimulus weights
        R           % independent noise variances
        T           % # time points per trial
        N           % # trials
        p           % # unobserved factors
        q           % # neurons
    end
    
    properties (Access = private)
        logLike     % log-likelihood curve during fitting
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
            p.addOptional('Tau', 5);
            p.addOptional('SigmaN', 1e-3);
            p.addOptional('Seed', 1);
            p.addOptional('Tolerance', 0.0005);
            p.parse(varargin{:});
            self.params = p.Results;

            % covariance function
            tau = self.params.Tau;
            sn = self.params.SigmaN;
            sf = 1 - sn;
            self.k = @(s, t) sf * exp(-1/2 * bsxfun(@minus, s, t) .^ 2 / tau ^ 2) + sn * bsxfun(@eq, s, t);
        end
        
        
        function self = fit(self, Y)
            % Fit the model
            %   self = fit(self, Y) fits the model to data Y.
            %
            %   See GPFA for optional parameters to use for fitting.
            
            self.runtime = now();
            
            % make sure dimensions of input are correct
            % TODO
            
            % ensure deterministic behavior
            rng(self.params.Seed);
            
            % initialize model using factor analysis
            % TODO
            
            % run EM
            self = self.EM();

            % output run time
            self.runtime = (now() - self.runtime) * 24 * 60 * 60; % convert to sec
            fprintf('Total run time: %.1f sec.\n\n\n', self.runtime)
        end
    end
    
    
    methods (Access = protected)
        
        function [Y, C, D, R] = expand(self)
            Y = self.Y;
            C = self.C;
            D = self.D;
            R = self.R;
        end
        
        
        function self = collect(self, Y, C, D, R)
            self.Y = Y;
            self.C = C;
            self.D = D;
            self.R = R;
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
            [Y, C, D, R] = expand(self);
            
            % pre-compute GP covariance and inverse
            p = self.p;
            q = self.q;
            T = self.T;
            N = self.N;
            t = 1 : T;
            K = self.k(t, t');
            Ki = inv(K);
            Kb = kron(K, eye(p));
            Kbi = kron(Ki, eye(p));
            S = eye(T);
            Yn = reshape(Y, [q T N]);
            
            iter = 0;
%             logLikeBase = NaN;
%             while iter < maxIter && (iter < 2 || (self.logLike(end) - self.logLike(end - 1)) / (self.logLike(end - 1) - logLikeBase) > self.params.Tolerance)
            while iter < maxIter
                
                iter = iter + 1;

                % Perform E step
                RiC = bsxfun(@rdivide, C, diag(R));
                CRiC = C' * RiC;
                Mi = invPerSymm(Kbi + kron(eye(T), CRiC), p);
                RbiCb = kron(eye(T), RiC); % [TODO] can kron be optimized?
                Rbi = kron(eye(T), diag(1 ./ diag(R)));
                Cb = kron(eye(T), C);
                KbCb = Kb * Cb'; % [TODO] optimize: K is diagonal
                Q = KbCb * (Rbi - RbiCb * Mi * RbiCb');
                YDS = bsxfun(@minus, Y, D * S);
                Ex = Q * reshape(YDS, q * T, N);
                Ex = reshape(Ex, [p T N]);
                Exx = Kb - Q * KbCb';
                
                % Perform M step
                T1 = zeros(q, p + T);
                T2 = zeros(p + T);
                for t = 1 : T
                    tt = (1 : p) + p * (t - 1);
                    for n = 1 : N
                        x = Ex(:, t, n);
                        s = S(:, t);
                        % [TODO] make more efficient: s mostly zero
                        T1 = T1 + Yn(:, t, n) * [x', s'];
                        T2 = T2 + [Exx(tt, tt) + x * x', x * s'; s * x', s * s'];
                    end
                end
                CD = T1 / T2;
                C = CD(:, 1 : p);
                D = CD(:, p + (1 : T));
                
                R = diag(mean(YDS .^ 2, 2) - ...
                    sum(bsxfun(@times, YDS * reshape(Ex, p, T * N)', C), 2) / (T * N));

                % calculate log-likelihood
                % TODO
                
%                 if iter == 1
%                     logLikeBase = self.logLike(end);
%                 end
            end
        end
        
    end
    
    
    methods (Static)
        
        function [gpfa, X] = toyExample()
            % Create toy example for testing
            
            rng(1);
            N = 20;
            T = 5;
            p = 2;
            q = 8;

            phi = cumsum(randn(1, T * N));
            phi = filtfilt(gausswin(5) / 2, 1, phi);
            X = [cos(phi); sin(phi)];
            
            phi = (0 : q - 1) / q * 2 * pi;
            C = [cos(phi); sin(phi)]';
            D = rand(q, T);
            S = repmat(eye(T), 1, N);
            R = 0.02 * eye(q);
            Y = chol(R)' * randn(q, T * N) + C * X + D * S;
            
            gpfa = collect(GPFA, Y, S, D, R);
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
