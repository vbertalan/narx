function [dJdA,dJdB]=grad(X,A,B,saida,h)

N=length(X);
Z=[X,ones(N,1)]*A';
V=tanh(Z);
Y=[V,ones(N,1)]*B';
erro=Y-saida;

dJdB=erro'*[V,ones(N,1)];
sig=erro*B(:,1:h).*(1-V.*V);
dJdA=sig'*[X,ones(N,1)];

