function [Ew,gradEw1,gradEw2] = cost_outputgrad_hiddengrad(W1,W2,X,T,lambda)

K = size(W2,1);

A = X*W1';
Z = cos(A);
Z = [ones(size(Z,1),1),Z];
Z1 = -sin(A);
Z1 = [ones(size(Z1,1),1),Z1];
Yx = Z*W2';

M = max(Yx,[],2);

W = W2(:,2:end)*W1;

Ew = sum(sum(T.*Yx)) - sum(M) - sum(log(sum(exp(Yx-repmat(M,1,K)),2))) - (0.5*lambda)*sum(sum(W.*W));


if nargout>1
   
    S = softmax(Yx);
    
    gradEw1 = (((T-S)*W2(:,2:end).*Z1(:,2:end))')*X - lambda*W1;
    
    gradEw2 = ((T-S)')*Z - lambda*W2;
    
end