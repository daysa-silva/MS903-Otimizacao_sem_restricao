# Problema: Encontrar os parâmetros x1, x2, x3 em que (x1,  x2) é o centro de um círculo de raio x3 
import numpy as np
from time import time_ns
from utils import exportarDados, norma_inf

def phi(x, k = 1):
    match k:
        case 1:
            f = x
            g = 1
        case 2:
            aux = 2 / np.pi
            
            f = aux * np.tanh(x)
            g = aux / (np.cosh(x)**2)
        case 3:
            f = np.tanh(x)
            g = 1/(np.cosh(x)**2)
        case 4:
            aux = np.sqrt(x**2 + 1)
            f = (x + aux)/2
            g = (1 + x/aux)/2

    return f, g

def f_aux(x, k1 = 2, k2 = 2):
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

    w_pred, _ = f_aux(x)
    w_pred = np.sign(w_pred)
    print('\nAcurácia Treino:', np.sum(w_pred == w)/w_pred.size)
    
    return x

def minimizador(x, erro, M, sigma=1/2):
    k = 0
    f_k, g_k = funcao_perda(x, g=True)
    
    while np.max(np.abs(g_k)) >= erro and k < M:
        d = - g_k
        t = 1

        while funcao_perda(x+t*d) >= f_k:
            t = t*sigma
        
        x = x + t*d
        f_k, g_k = funcao_perda(x, g=True)
        k = k + 1
    
    print('Iterações:',k )
    
    w_pred, _ = f_aux(x)
    w_pred = np.sign(w_pred)
    print('\nAcurácia Treino:', np.sum(w_pred == w)/w_pred.size)
    
    return x

# Dados de treino
u, v, w = exportarDados('../Trabalho/circulo_treina')

# Encontrar o vetor x = [x1, x2, x3] que minimiza a função perda
x_inicial = np.array([1,-2, 4, 0, -3, 1, 0, 5, 0.5])
erro = 10e-7
M = 10000

inicio = time_ns()

## DESCOMENTAR DE ACORDO COM O MÉTODO DESEJADO
x_otimo = minimizador_nao_monotono(x_inicial, erro, M, sigma=0.2, alpha=10e-4, historico_passos=1)
#x_otimo = minimizador(x_inicial, erro, M, sigma=0.3)

fim = time_ns()

tempo_execucao = (fim - inicio) * 1e-9
print('\nTempo de execução %.4f' % tempo_execucao)
print('\nx* =', x_otimo, '\n')

## PERFORMANCE NO CONJUNTO DE TESTE
# Dados de teste
u, v, w = exportarDados('../Trabalho/circulo_testa')

w_pred, _ = f_aux(x_otimo)
w_pred = np.sign(w_pred)

print('Acurácia Teste:', np.sum(w_pred == w)/w_pred.size)