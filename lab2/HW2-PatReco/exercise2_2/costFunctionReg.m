function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
J = ( 1/m ) * sum(Cross_entr)  + ( lambda/(2*m) )*sum( theta(2:size(theta,1),1).^2 );

% gradient of J
grad(1,1) = (  (1/m)*sum( (-y + h).*X(:,1) )  );

for j = 2:size(theta,1)
    grad(j,1) = ( 1/m ) * ( sum( (-y + h) .* X(:,j) ) + lambda*theta(j,1) );
end

%disp(J)
%disp(size(grad))

% =============================================================

end
