function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================

% logistical estimate of y is y_hat_i = h( x_i )
h = sigmoid( X*theta );

% cross entropy
Cross_entr = -y.*log(h) - (1-y).*log(1-h);

% cost function J
J = ( 1/m ) * sum(Cross_entr);

% gradient of J
for j = 1:size(theta,1)
    grad(j) = ( 1/m ) * sum( (-y + h) .* X(:,j) );
end


% =============================================================

end
