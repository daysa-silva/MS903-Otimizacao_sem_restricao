# Problema: Encontrar os parâmetros x1, x2, x3 em g(u,x) = x1  + x2 * exp(x3*u)
import numpy as np
import matplotlib.pyplot as plt
from time import time_ns
from utils import exportarDados, norma_inf

def funcao_perda(x, g=False):
    N = u.size

    aux1 = x[0] + x[1]*u - v
    aux2 = np.arctan(aux1) + np.pi*w/2
    aux3 = 1 + aux1**2

    f = np.sum(aux2*aux2) / (2*N)

    if g:
        g =  np.array([np.sum(aux2 / aux3),  np.sum(aux2 *u / aux3)]) / N
        return f, g
    else:
        return f
    
def minimizador_nao_monotono(x0, erro, M, alpha=10e-4, sigma=1/2, historico_passos=5):
    k = 0
    x = x0
    f_k, g_k = funcao_perda(x0, g=True)

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

    
# Dados de treino
u, v, w = exportarDados('plano2.txt')

# Dados de teste
u_teste, v_teste, w_teste = exportarDados('plano.txt')

# Encontrar o vetor x = [x1, x2, x3] que minimiza a função perda
x_inicial = np.array([0,0])
erro = 1e-5
M = 10000

inicio = time_ns()
x_otimo = minimizador_nao_monotono(x_inicial, erro, M, sigma=0.5)
fim = time_ns()

tempo_execucao = (fim - inicio) * 1e-9
print('\nTempo de execução %.4f' % tempo_execucao)
print('\nx* =', x_otimo, '\n')

## Data for plotting
plt.title('Problema de Classificação')
# Elementos da classe - (w[j] == -1)
plt.plot(u_teste[w_teste < 0],v_teste[w_teste < 0], '.b')
# elementos da classe + (w[j] == +1)
plt.plot(u_teste[w_teste > 0],v_teste[w_teste > 0], '.r')

# Plano que separa as classes
t = np.array([-10,10])
plt.plot(t, x_otimo[0] + x_otimo[1]*t)

plt.show()
