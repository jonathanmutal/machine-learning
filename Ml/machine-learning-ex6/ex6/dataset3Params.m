function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
    values = [0.01 0.03 0.1 0.3 1 3 10 30];

    error = inf;

    for sig = values
        for c = values
            model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
            predictions = svmPredict(model, Xval);

            e = mean(double(predictions ~= yval));
            if e < error
                error = e;
                C = c;
                sigma = sig;
            end
        end
    end
end
