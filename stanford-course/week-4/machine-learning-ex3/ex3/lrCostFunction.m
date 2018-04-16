function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% The main goal here is to calcualte the cost and the vector of with regularization

% Pre calculate sigmoid of the Decision Boundary, because we will use ir a lot
s = sigmoid(X*theta);

% This is a bit tricky: since we don't want to use regularization for the 
% unit bias (the first parameter theta_0), we build a new theta vector with 0
% as the first value and then the rest. We do this in order to calculate in a
% vectorized way, the cost and gradient (X_0 = 1 will multiply theta_0 
% (that will be 0)).
helper_theta = [0; theta(2:end)];

% Now we calculate J and we sum and element-wise pow to 2, because first element
% of helper_theta is 0
J = - (1 / m) * (y' * log(s) + (1 - y)' * log(1 - s)) + (lambda/(2*m))*sum(helper_theta .^2);
% We calculate gradient as well
grad = (1 / m) .* (X' * (s - y)) + (lambda/m) * helper_theta;







% =============================================================

grad = grad(:);

end
