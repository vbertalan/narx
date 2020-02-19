import numpy

# CLASSE QUE IMPLEMENTA UMA REDE DE RECORRÊNCIA EXTERNA
class RedeNARX(object):
    def __init__(self, oculta_tamanho=3, taxa_aprendizado=0.2, max_epocas=1000, atrasos=2):
        self.oculta_tamanho = oculta_tamanho
        self.taxa_aprendizado = taxa_aprendizado
        self.max_epocas = max_epocas
        self.atrasos = atrasos
        self.auc = 0.5

 # TREINA A REDE DE RECORRÊNCIA EXTERNA, COM BASE NOS VALORES DE TAMANHO DEFINIDOS COMO ARGUMENTOS
 def treinamento(self, X, y):
       
        self.entrada_tamanho = X.shape[1]
        self.saida_tamanho = y.shape[1]
        epocas_faltantes = self.max_epocas

        # Inicializa pesos

        self.W1 = numpy.random.rand(1 + self.entrada_tamanho, self.oculta_tamanho)
        self.W2 = numpy.random.rand(1 + self.oculta_tamanho, self.saida_tamanho)
        self.W3 = numpy.random.rand(self.saida_tamanho * self.atrasos, self.oculta_tamanho)
        self.Yatrasado = numpy.zeros((1, self.saida_tamanho * self.atrasos))

        epsilon = 0.001
        erro = 1
        # Erro
        self.J = []         

        # Repete até que o erro seja muito pequeno, ou max_epocas seja alcancado

        while erro > epsilon and epocas_faltantes > 0:
            total_erro = numpy.array([])

            # Para cada época
            for self.X, self.y in zip(X, y):
                self.X = numpy.array([self.X])
                self.y = numpy.array([self.y])
                erro, gradientes = self.passo(self.X, self.y)
                total_erro = numpy.append(total_erro, erro)
                dJdW1 = gradientes[0]
                dJdW2 = gradientes[1]
                dJdW3 = gradientes[2]

                # Calcula novos pesos
                self.W1 = self.W1 - self.taxa_aprendizado * dJdW1
                self.W2 = self.W2 - self.taxa_aprendizado * dJdW2
                self.W3 = self.W3 - self.taxa_aprendizado * dJdW3

                # Muda Yatrasado a cada época
                self.Yatrasado = numpy.roll(self.Yatrasado, 1, 1)
                self.Yatrasado[:,::self.atrasos] = self.Y

            epocas_faltantes -= 1
return self

# APLICA O MÉTODO DE TREINAMENTO, PASSO A PASSO
def passo(self, X, y):
        self.Y = self.forward(X)
        custo = self.custo(self.Y, y)
        gradientes = self.backpropagate(X, y)
return custo, gradientes

# JOGA OS VALORES DE ENTRADA PARA FRENTE NA REDE, E RETORNA OS VALORES DE SAÍDA 
# CAMADA OCULTA É SIGMÓIDE, CAMADA DE SAÍDA É LINEAR
def forward(self, X):
        self.Zin = numpy.dot(X, self.W1[:-1,:])
        self.Zin += numpy.dot(numpy.ones((1, 1)), self.W1[-1:,:])
        self.Zin += numpy.dot(self.Yatrasado, self.W3)
        self.Z = self.sigmoide(self.Zin)
        self.Z = numpy.nan_to_num(self.Z)

        self.Yin = numpy.dot(self.Z, self.W2[:-1,])
        self.Yin += numpy.dot(numpy.ones((1, 1)), self.W2[-1:,:])
        Y = self.linear(self.Yin)
        Y = numpy.nan_to_num(Y)
return Y

# BACKPROPAGA OS CUSTOS ATRAVÉS DA REDE
def backpropagate(self, X, y):
        delta3 = numpy.multiply(-(y-self.Y), self.derivada_linear(self.Yin))
        dJdW2 = numpy.dot(self.Z.T, delta3)
        dJdW2 = numpy.append(dJdW2, numpy.dot(numpy.ones((1, 1)), delta3), axis=0)

        delta2 = numpy.dot(delta3, self.W2[:-1,:].T)*self.derivada_sigmoide(self.Zin)
        dJdW1 = numpy.dot(X.T, delta2)
        dJdW1 = numpy.append(dJdW1, numpy.dot(numpy.ones((1, 1)), delta2), axis=0)

        dJdW3 = numpy.dot(numpy.repeat(self.Yatrasado, self.saida_tamanho * self.atrasos, 0), \
                          numpy.repeat(delta2, self.saida_tamanho * self.atrasos, 0))
return dJdW1, dJdW2, dJdW3

# ATIVAÇÃO SIGMÓIDE
def sigmoide(self, z):
        return 1/(1+numpy.exp(-z))

# DERIVADA SIGMÓIDE
def derivada_sigmoide(self, z):
        return numpy.exp(-z)/((1+numpy.exp(-z))**2)

# ATIVAÇÃO LINEAR
def linear(self, z):
        return z

# DERIVADA LINEAR (= 1)
def derivada_linear(self, z):
        return 1

# CALCULA ERRO QUADRÁTICO MÉDIO 
def custo(self, Y, y):        
		return mean_squared_error(Y, y)