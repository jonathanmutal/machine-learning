function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
pass = find(y == 1); nPass = find(y == 0);

plot(X(pass, 1), X(pass, 2), 'k+', 'linewidth', 2, 'markerSize', 7);
plot(X(nPass, 1), X(nPass, 2), 'ko', 'markerFacecolor', 'y', 'markerSize', 7);

% =========================================================================



hold off;

end
