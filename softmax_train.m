function [W1,W2] = softmax_train(Winit1,Winit2,X,T,lambda)

W1 = Winit1;
W2 = Winit2;

Ewold = -Inf;
eta = (0.1/size(T,1));
for iteration=1:500
    [Ew, gradEw1, gradEw2] = cost_outputgrad_hiddengrad(W1,W2,X,T,lambda);
    
    if(abs(Ew-Ewold)<(1e-6))
        break;
    end
    
    W1 = W1+eta*gradEw1;
    
    W2 = W2+eta*gradEw2;
    
    Ewold = Ew;
end