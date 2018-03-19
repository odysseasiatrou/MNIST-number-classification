clear all;
close all;

load mnist_all.mat;

diary('results(M=200).txt');

K = 10;

M = 200;
T = [];
X = [];
Ttesttrue = [];
Xtest = [];
Ntrain = zeros(1,10);
Ntest = zeros(1,10);
for j=1:10
    s = ['train' num2str(j-1)];
    Xtmp = double(eval(s));
    Ntrain(j) = size(Xtmp,1);
    Ttmp = zeros(Ntrain(j),K);
    Ttmp(:,j) = 1;
    X = [X; Xtmp];
    T = [T; Ttmp];
    
    s = ['test' num2str(j-1)];
    Xtmp = double(eval(s));
    Ntest(j) = size(Xtmp,1);
    Ttmp = zeros(Ntest(j),K);
    Ttmp(:,j) = 1;
    Xtest = [Xtest; Xtmp];
    Ttesttrue = [Ttesttrue; Ttmp];
    
end

X = X/255;
Xtest = Xtest/255;

[N D] = size(X);

X = [ones(sum(Ntrain),1), X];
Xtest = [ones(sum(Ntest),1), Xtest];
 
lambda = 0;    

Winit1 = zeros(M,D+1);
Winit2 = zeros(K, M+1);

W1 = randn(size(Winit1));
W2 = randn(size(Winit2));

ch = randperm(N); 
ch = ch(1:20);
[gradEw1,numgradEw1,gradEw2,numgradEw2] = grad_check(W1,W2,X(ch,:),T(ch,:),lambda);

Winit1 = randn(M,D+1)/sqrt(M);
[W1,W2] = softmax_train(Winit1, Winit2, X, T, lambda);

[Ttest, Ytest] = softmax_test(W1, W2, Xtest);

[~, Ttrue] = max(Ttesttrue,[],2); 
err = length(find(Ttest~=Ttrue))/10000;
disp(['The error of the method is: ' num2str(err)])

diary('off');