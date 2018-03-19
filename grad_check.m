function [gradEw1,numgradEw1,gradEw2,numgradEw2] = grad_check(W1,W2,X,T,lambda)

[Ew, gradEw1, gradEw2] = cost_outputgrad_hiddengrad(W1,W2,X,T,lambda);

[M D] = size(W1);

epsilon = 1e-6;
numgradEw1 = zeros(M,D);
for m=1:M
    for d=1:D
        Wtmp = W1;
        Wtmp(m,d)= Wtmp(m,d)+epsilon;
        Ewplus = cost_outputgrad_hiddengrad(Wtmp,W2,X,T,lambda);
        
        Wtmp = W1;
        Wtmp(m,d) = Wtmp(m,d)-epsilon;
        Ewminus = cost_outputgrad_hiddengrad(Wtmp,W2,X,T,lambda);
        
        numgradEw1(m,d) = (Ewplus-Ewminus)/(2*epsilon);
    end
end

[K M] = size(W2);
numgradEw2 = zeros(K,M);
for k=1:K
    for m=1:M
        Wtmp=W2;
        Wtmp(k,m)=Wtmp(k,m)+epsilon;
        Ewplus = cost_outputgrad_hiddengrad(W1,Wtmp,X,T,lambda);
        
        Wtmp=W2;
        Wtmp(k,m)=Wtmp(k,m)-epsilon;
        Ewminus =cost_outputgrad_hiddengrad(W1,Wtmp,X,T,lambda);
        
        numgradEw2(k,m)=(Ewplus-Ewminus)/(2*epsilon);
    end
end

diff1=abs(numgradEw1-gradEw1);

diff2=abs(numgradEw2-gradEw2);

disp(['The maximum absolute norm for the hidden layer in the gradcheck is ' num2str(max(diff1(:))) ]);
disp(['The maximum absolute norm for the output layer in the gradcheck is ' num2str(max(diff2(:))) ]);
