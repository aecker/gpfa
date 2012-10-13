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
        S           % stimulus predictors
        k           % GP covariance function
        C           % factor loadings
        D           % stimulus weights
        R           % independent noise variances
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
        
        
        function self = fit(self, Y, S)
            % Fit the model
            %   self = fit(self, Y, S) fits the model to data Y using
            %   stimulus predictors S.
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
        
        function [Y, S, C, D, R] = expand(self)
            Y = self.Y;
            S = self.S;
            C = self.C;
            D = self.D;
            R = self.R;
        end
        
        
        function self = collect(self, Y, S, C, D, R)
            self.Y = Y;
            self.S = S;
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
            iter = 0;
            logLikeBase = NaN;
            while iter < maxIter && (iter < 2 || (self.logLike(end) - self.logLike(end - 1)) / (self.logLike(end - 1) - logLikeBase) > self.params.Tolerance)
                
                if ~mod(iter, 5), fprintf('.'), end
                iter = iter + 1;

                % Perform E step

                % Perform M step
                
                % calculate log-likelihood
            
                if iter == 1
                    logLikeBase = self.logLike(end);
                end
            end
        end
        
    end
    
end
