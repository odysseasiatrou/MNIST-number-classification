function [Ttest, Ytest] = softmax_test(W1, W2, Xtest)

A = Xtest*W1';
Z = cos(A);
Z = [ones(size(Z,1),1),Z];
Ytest = softmax(Z*W2');

[~, Ttest] = max(Ytest, [], 2);