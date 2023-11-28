# Problema: Encontrar os parâmetros x1, x2, x3 em que (x1,  x2) é o centro de um círculo de raio x3 
import numpy as np
import matplotlib.pyplot as plt
from time import time_ns
from utils import exportarDados, norma_inf

def r(x,a,b):
    return (x[0] - a)**2 + (x[1] - b)**2 - x[2]**2

def funcao_perda(x, g=False):
    N = a.size
    aux1 = r(x,a,b)
    aux2 = np.arctan(aux1) - np.pi*c/2
    aux3 = 1 + aux1**2

    f = np.sum(aux2*aux2) / (2*N)

    if g:
        g =  np.array([np.sum(aux2 / aux3 * 2*(x[0] - a)),  np.sum(aux2 / aux3 * 2*(x[1] - b)), np.sum(aux2 / aux3 * (-2)*x[2])]) / N
        return f, g
    else:
        return f
    
def minimizador_nao_monotono(x0, erro, M, alpha=10e-4, sigma=1/2, historico_passos=5):
    k = 0
    x = x0
    f_k, g_k = funcao_perda(x0, g=True)
    print(g_k)

    lista_f_x = []

    while norma_inf(g_k) >= erro and k < M:
        # Atualizar a lista_f_x
        if len(lista_f_x) == historico_passos:
            lista_f_x.pop(0)
        lista_f_x.append(f_k)

        #Direção de Descida
        d = - g_k
        
        if  k == 0:
            t = 1
        else:
            t = np.linalg.norm(x - x_anterior) / np.linalg.norm(g_k - g_k_anterior)
            
        while funcao_perda(x + t * d) > np.max(lista_f_x) + alpha * t * np.dot(g_k, d):
            t = sigma*t
        
        # Atualizar x
        x_anterior = x
        x = x_anterior + t*d

        # Atualizar g_k
        g_k_anterior = g_k
        f_k, g_k = funcao_perda(x, g=True)

        k = k + 1
    
    print('\nIterações:', k)
    print('Função perda:', f_k)
    
    return x

def minimizador(x, erro, M):
    k = 0
    f_k, g_k = funcao_perda(x, g=True)
    
    while np.max(np.abs(g_k)) >= erro and k < M:
        d = - g_k
        t = 1

        while funcao_perda(x+t*d) >= f_k:
            t = t/2
        
        x = x + t*d
        f_k, g_k = funcao_perda(x, g=True)
        k = k + 1
    
    print('Iterações:',k )
    print('Função perda:', f_k)
    
    return x

# Dados de treino
a, b, c = exportarDados('circulo_treina')

# Dados de teste
a_teste, b_teste, c_teste = exportarDados('circulo_testa')

# Encontrar o vetor x = [x1, x2, x3] que minimiza a função perda
x_inicial = np.array([2,-2, 3])
erro = 10e-2
M = 10000

inicio = time_ns()

## DESCOMENTAR DE ACORDO COM O MÉTODO DESEJADO
#x_otimo = minimizador_nao_monotono(x_inicial, erro, M, sigma=0.2, alpha=10e-2, historico_passos=5)
#x_otimo = minimizador(x_inicial, erro, M)

fim = time_ns()

tempo_execucao = (fim - inicio) * 1e-9
print('\nTempo de execução %.4f' % tempo_execucao)
print('\nx* =', x_otimo, '\n')

## Data for plotting
figure, axes = plt.subplots() 

plt.title('Circulo Separador')
# Elementos da classe - (w[j] == -1)
axes.plot(a_teste[c_teste < 0],b_teste[c_teste < 0], '.b')
# elementos da classe + (w[j] == +1)
axes.plot(a_teste[c_teste > 0],b_teste[c_teste > 0], '.r')

# Plano que separa as classes
Drawing_uncolored_circle = plt.Circle( (x_otimo[0], x_otimo[1] ), x_otimo[2] ,  fill = False ) 
  
axes.set_aspect( 1 ) 
axes.add_artist( Drawing_uncolored_circle ) 

plt.show()
