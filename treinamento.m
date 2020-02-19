%**************************************************************
% Este programa realiza o treinamento de uma rede perceptrons
% com uma camada intermediaria
% Metodo do gradiente
% Treinamento em batelada
% Data: 22/06/99
% Autor: Clodoaldo Aparecido de Moraes Lima
%****************************************************************

function [A,B,veterro_tr]=treinamento(X,S,A,B,h,nepocasmax)

% Calcula o numero de entradas e Ss
[N,ne]=size(X);
[N,ns]=size(S);

if isempty(A)&isempty(B)
    disp('Inicializando a rede neural')
    A=rands(h,ne+1)/5;
    B=rands(ns,h+1)/5;
end


% Define a taxa de aprendizado
alfa=1;

%Define o erro minimo
erromin=1e-5;

% Define o numer de epocas maximo
%nepocasmax=250;

% Calcula a S da rede
Sr=perceptrons(X,A,B);

% Calcula o vetor com os erros
erro=Sr-S;

% Calcula o Erro quadratico medio
EQM=(1/N)*(sum(sum(erro.*erro)));
veterro_tr=[];
veterro_tr=[veterro_tr,EQM];
nepocas=0;
disp('Treinando a Rede Neural. Aguarde .....')
while  EQM>erromin & nepocas <nepocasmax
    
    % Incrementa o numero de epocas
    nepocas=nepocas+1;
    
    % Calcula o Gradiente
    [dJdA,dJdB]=grad(X,A,B,S,h);
    
    % Transforma para vetor
    vetdJdA=reshape(dJdA,1,h*(ne+1))';
    vetdJdB=reshape(dJdB,1,ns*(h+1))';
    
    % Monta o vetor gradiente
    g=[vetdJdA;vetdJdB];
    
    % Normaliza o vetor gradiente
    g=g/norm(g);
    
    % Transforma em vetor
    vetdJdA=g(1:h*(ne+1));
    vetdJdB=g(h*(ne+1)+1:end);
    
    % Transforma em matriz
    dJdA=reshape(vetdJdA',h,ne+1);
    dJdB=reshape(vetdJdB',ns,h+1);
    
    % Atualiza a matriz A e B
    Aatual=A-alfa*dJdA;
    Batual=B-alfa*dJdB;
    
    % Calcula a S da rede
    Sr=perceptrons(X,Aatual,Batual);
    
    % Calcula o vetor com os erros
    erro=Sr-S;
    
    % Calcula o Erro quadratico medio
    EQMatual=(1/N)*(sum(sum(erro.*erro)));
    while EQMatual>EQM
        alfa=alfa*0.9;
        
        % Atualiza a matriz A e B
        Aatual=A-alfa*dJdA;
        Batual=B-alfa*dJdB;
        
        % Calcula a S da rede
        Sr=perceptrons(X,Aatual,Batual);
        
        % Calcula o vetor com os erros
        erro=Sr-S;
        
        % Calcula o Erro quadratico medio
        EQMatual=(1/N)*(sum(sum(erro.*erro)));
    end
    
    % Incrementa a taxa de aprendizagem
    alfa=alfa/0.9;
    A
    Aatual
    alfa
    save all
    % Atualiza as matrizes de entrada e S
    A=Aatual;
    B=Batual;
    EQM=EQMatual;	
    veterro_tr=[veterro_tr,EQM];
end



figure(1)
plot(veterro_tr)


