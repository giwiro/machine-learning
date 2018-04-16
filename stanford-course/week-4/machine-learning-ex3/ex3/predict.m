function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%
%       can use max(A, [], 2) to obtain the max for each row.

% We add the unit bias to the initial X array
% X -> m x n
% Append bunch of ones
A_1 = [ones(m, 1) X]
% A -> m x (n + 1)

% We calculate the first Z. To do that we multiply A_1 with Theta1 traspose
% in order to multiply each theta with it's corresponding x for each 
% row (training example). Wel will get Z_2.
% Theta1 -> K x (n + 1)
Z_2 = A_1 * Theta1';
% Z_2 -> m x K
% We add the unit bias to A_2
A_2 = [ones(m, 1) sigmoid(Z_2)];
% A_2 -> m x (K + 1)

% Calculate Z_3.
% Theta2 -> 1 x (K + 1)
Z_3 = A_2 * Theta2';
% Z_3 -> m x 1
% In this last one, we won't append ones because is the final answer
A_3 = sigmoid(Z_3);
% A_3 -> m x 1

% We extract the index of the maximum value for each row
[pv, pi] = max(A_3, [], 2);
p = pi;


% =========================================================================


end
