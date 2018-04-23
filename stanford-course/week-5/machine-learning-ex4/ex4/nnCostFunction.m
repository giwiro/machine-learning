function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% s = 25
% n = 400
% K = 10

% Theta1 --> s x (n + 1)
% Theta2 --> K x (s + 1)

% Add unit bias to create A_1
% A_1 --> m x (n +1)
A_1 = [ones(m, 1), X];
% We calculate Z_2
% Z_2 --> m x s
Z_2 = A_1 * Theta1';
% We calculate A_2 by using the sigmoid func over Z_2
% We are adding one column of one's at the start because the unit bias
% since we are in forward prop. we include the unit bias for A_2 for later
% calculate Z_3 (with the unit bias)
% A_2 --> m x (s + 1)
A_2 = [ones(m, 1), sigmoid(Z_2)];
% Calculate Z_3
% Z_3 --> m x K
Z_3 = A_2 * Theta2';
% We calculate A_3 by applying the sigmoid to Z_3
% No unit bias addition for last layer (it's also hypotheses func.)
% A_3 --> m x K
A_3 = h = sigmoid(Z_3);
% Since we got a vector of labels, we need to transform that into a matrix
% yv --> m x K
yv = bsxfun(@eq, y, 1:num_labels);
% Now we calculate the cost in a vectorized way
% cost --> m x K
cost = (yv).*log(h) + (1 - yv).*log(1 - h);
% Since it's a matrix and we need scalar value, we apply sum over all elements
% J --> 1 x 1
J = (-1/m)*sum(sum(cost));

% Regularization, but just for cost function (Not backpropagation yet !!)
% Trim first column for both Theta's matrices for later add it up with J
t1 = Theta1(:, 2:end);
t2 = Theta2(:, 2:end);

% Calculate regularization value.
reg = (lambda / (2 * m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2)));

% Add regularization to J
J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% The idea right here is to obtain the gradient (derivative) of the cost function
% In order to do that we must:
%   1. Do fwd. propagation (to calculate all a's and z's)
%   2. Calculate all the deltas for each layer (3 and 2)
%   3. With the a's and deltas calculated, we build the Delta's matrices
%   4. Regularization (later)
%   5. Loot

% Initialize Delta's matrices
% Delta1 --> s x (n + 1) Same as Theta1
Delta1 = zeros(size(Theta1));
% Delta2 --> K x (s + 1) Same as Theta1
Delta2 = zeros(size(Theta2));

% For now we are going to do a for statement.
for t = 1:m,
  % 1. Forward propagation part
  
  % We trim the A_1 matrix to only get one row and then transform into column
  % it's more easy and readable.
  % a1t --> (n + 1) x 1
  a1t = A_1(t,:)';
  % We calculate z2t
  % z2t --> s x 1
  % ^ Note that it's length is (s x 1) it doesn't matter for fwd. propagation,
  %   but for back propagation (since it's backwards) it MUST include the unit bias
  z2t = Theta1 * a1t;
  % We calculate a2t by applying the sigmoid func. to z2t and then add a 1 for
  % unit bias
  % a2t --> (s + 1) x 1
  a2t = [1; sigmoid(z2t)];
  % We calculate z3t
  % z3t --> K x 1
  % ^ We dont't care fo unit bias since we won't be using it for backpropagation
  %   because it's the last layer and it doesn't has unit bias
  z3t = Theta2 * a2t;
  % We finally get out hypotheses func. by applying sigmoid func. to z3t
  % awt --> K x 1
  a3t = ht = sigmoid(z3t);
  % We trim our yv matrix to get one vector corresponding to the row selected.
  % We also traspose it, cos we are working with columns for vectors.
  % yt --> K x 1
  yt = yv(t, :)';
  
  % 2. Calculate deltas
  
  % Calculate the d3t (last layer delta) is the easiest
  % d3t --> K x 1
  d3t = ht - yt;
  % We add the bias unit to z2t cos we are in backpropagation.
  % Another way to see this is: if we don't add the bias unit it won't fit
  % for calculating d2t.
  % z2t --> (s + 1) x 1
  z2t = [1; z2t];
  % Now we calculate d2t using the formula
  % d2t --> (s + 1) x 1
  d2t = (Theta2' * d3t).*sigmoidGradient(z2t);
  % Now that we calculated d2t, we no longer need the extra bias, so we trim
  % the first column, remember that delta must be same size as a for each layer
  % with no unit bias.
  % d2t --> s x 1
  d2t = d2t(2:end);
  
  % Now we calculate both Theta's with the formula
  % Delta1 --> s x (n + 1)
  Delta1 = Delta1 + d2t * a1t';
  % Delta2 --> K x (s + 1)
  Delta2 = Delta2 + d3t * a2t';
endfor;

% We will comment this section in order to  implement it taking care of
% the unit bias later in Part 3
  
%Theta1_grad = (1 / m) * Delta1;
%Theta2_grad = (1 / m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% First we divide just the first column of Theta's by m 
% Remember the formula
Theta1_grad(:, 1) = Delta1(:, 1) ./ m;
Theta2_grad(:, 1) = Delta2(:, 1) ./ m;

% Then we use the regularization from column 2 to the end.
Theta1_grad(:, 2:end) = (Delta1(:, 2:end) ./ m) + (lambda/m).*Theta1(:,2:end);
Theta2_grad(:, 2:end) = (Delta2(:, 2:end) ./ m) + (lambda/m).*Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
