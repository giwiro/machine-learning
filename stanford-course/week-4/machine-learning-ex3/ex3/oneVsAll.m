function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

options = optimset('GradObj', 'on', 'MaxIter', 50);

% The main idea here is to generate a (k x n + 1) matrix that will hold the theta's
% for all the labels available

% Here we iterate between all the labels
for k = 1:num_labels;
  % Initialize the thetas as a (n + 1 x 1) matrix, it's n + 1 because we want to
  % include the unit bias, since we append a bunch of one's in the X matrix at
  % the first column
  initial_theta = zeros(n + 1, 1);
  % This auto-function 'fmincg' will calculate the optimal theta values. 
  % The @(t) means we will pass a parameter to our function lrCostFunction
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y == k), lambda)), initial_theta, options);
  % We got the theta result as a (n + 1 x 1) in order to fit it in our matrix
  % we will traspose it.
  all_theta(k,:) = theta';
endfor;










% =========================================================================


end
