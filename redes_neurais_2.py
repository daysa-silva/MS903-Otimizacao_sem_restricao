import numpy as np
from time import time_ns
import matplotlib.pyplot as plt
from utils import exportarDados, norma_inf

def phi(x, k = 1):
    match k:
        case 1:
            f = x
            g = 1
        case 2:
            aux = 2 / np.pi
            
            f = aux * np.arctan(x)
            g = aux / (1 + x*x)
        case 3:
            f = np.tanh(x)
            g = 1/(np.cosh(x)**2)
        case 4:
            aux = np.sqrt(x*x + 1)

            f = (x + aux)/2
            g = (1 + x/aux)/2

    return f, g

def f_aux(x, k1 = 4, k2 = 4):
    a1 = x[0]*u + x[1]*v + x[4]
    b1, c1 = phi(a1, k1)

    a2 = x[2]*u + x[3]*v + x[5]
    b2, c2 = phi(a2, k1)

    a = x[6]*b1 + x[7]*b2 + x[8]
    b, c = phi(a, k2)

    g = np.array([
        c*x[6]*c1*u,
        c*x[6]*c1*v,
        c*x[7]*c2*u,
        c*x[7]*c2*v,
        c*x[6]*c1,
        c*x[7]*c2,
        c*b1,
        c*b2,
        c
    ])

    return b, g

def funcao_perda(x, r = 0, g=False):
    N = u.size
    
    a, b = f_aux(x)
    aux = a - w

    f = np.sum(aux*aux) / (2*N) + r/2*(np.linalg.norm(x)**2)

    if g:
        g =  (b@aux / N) + r*x
        return f, g
    else:
        return f

def minimizador(x, erro, M, sigma=1/2, reg = 0):
    k = 0
    f_k, g_k = funcao_perda(x, r = reg, g=True)
    
    while norma_inf(g_k) >= erro and k < M:
        d = - g_k

        if  k == 0:
            t = 1
        else:
            t = np.linalg.norm(x - x_anterior) / np.linalg.norm(g_k - g_k_anterior)

        while funcao_perda(x+t*d, r = reg) >= f_k:
            t = t*sigma
        
        # Atualizar x
        x_anterior = x
        x = x_anterior + t*d

        # Atualizar g_k
        g_k_anterior = g_k
        f_k, g_k = funcao_perda(x, r = reg, g=True)

        k = k + 1

    print(f_k -  reg/2*(np.linalg.norm(x)**2))
    
    print('\nIterações:', k)

    return x

# Dados de treino
u, v, w = exportarDados('Dados/treino.txt')

## PARAMETROS INICIAIS
x_inicial = np.array([0,0,0,0,0,0,0,0,0])
erro = 10e-4
sigma = 0.3
M = 10000
reg = 100

inicio = time_ns()

x_otimo = minimizador(x_inicial, erro, M, reg = reg, sigma = sigma)

fim = time_ns()

# tempo_execucao = (fim - inicio) * 1e-9
# print('\nTempo de execução %.4f' % tempo_execucao)
# print('\nx* =', x_otimo, '\n')

w_pred, _ = f_aux(x_otimo)
erro = np.linalg.norm(w - w_pred) / np.linalg.norm(w) * 100
print('\nErro Treino (norma-2):', erro)

# ## PERFORMANCE NO CONJUNTO DE TESTE
# Dados de teste
u, v, w = exportarDados('Dados/teste.txt')

w_pred, _ = f_aux(x_otimo)
erro = np.linalg.norm(w - w_pred)  / np.linalg.norm(w) * 100
print('\nErro Teste (norma-2):', erro)

plt.plot(w, w_pred, 'o')
plt.show()