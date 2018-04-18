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

constant = -1.0 / m;
linear = X * theta;
prediction = sigmoid(linear);
disp(prediction);
SummationTerms = y' * log(prediction) + (1-y)' *log(1-prediction);
J = constant*sum(SummationTerms);


grad = (X' * (prediction - y))/m;








% =============================================================

end
