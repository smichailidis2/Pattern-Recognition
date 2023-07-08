function f = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
f = zeros(size(z));

% ====================== YOUR CODE HERE ======================

f = 1./(1+exp(-z));

% =============================================================

end
