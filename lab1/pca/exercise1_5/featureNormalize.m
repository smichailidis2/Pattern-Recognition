function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%initialization
mu      = zeros(1,size(X,2)); % row vector of zeroes
sigma   = zeros(1,size(X,2)); % row vector of zeroes
X_norm  = X;

for i = 1:size(X,2)
    mu(:,i)     = mean(X(:,i));
    sigma(:,i)  = std(X(:,i));
    X_norm(:,i) = (X_norm(:,i)-mu(1,i)) ./ sigma(1,i);
end
% ============================================================

end