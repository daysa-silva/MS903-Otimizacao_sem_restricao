import numpy as np
from utils import norma_inf

def minimizador_armijo(x, erro, M, alpha=10e-4, sigma=1/2):
    k = 0
    g_k = funcao_perda(x, g=True)
    
    while norma_inf(np.abs(g_k)) >= erro and k < M:
        d = - g_k
        t = 1
        
        while funcao_perda(x+t*d) > funcao_perda(x) + alpha * t * np.dot(g_k, d):
            t = sigma*t
        x = x + t*d
        g_k = funcao_perda(x, g=True)
        k = k + 1
    
    print('Iterações:', k)
    
    return x

def minimizador_armijo_modificado(x, erro, M, alpha=10e-4, sigma=1/2):
    k = 0
    f_k, g_k = funcao_perda(x, g=True)
    
    while norma_inf(g_k) >= erro and k < M:
        d = - g_k
        
        if  k == 0:
            t = 1
        else:
            t = np.linalg.norm(x - x_anterior) / np.linalg.norm(g_k - g_k_anterior)
            
        while funcao_perda(x + t * d) > f_k + alpha * t * np.dot(g_k, d):
            t = sigma*t
        
        # Atualizar x
        x_anterior = x
        x = x_anterior + t*d

        # Atualizar g_k
        g_k_anterior = g_k
        f_k, g_k = funcao_perda(x, g=True)
    
    print('Iterações:', k)
    
    return x

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
            
        while funcao_perda(x + t * d) > norma_inf(lista_f_x) + alpha * t * np.dot(g_k, d):
            t = sigma*t
        
        # Atualizar x
        x_anterior = x
        x = x_anterior + t*d

        # Atualizar g_k
        g_k_anterior = g_k
        f_k, g_k = funcao_perda(x, g=True)

        k = k + 1
    
    print('\nIterações:', k)

def minimizador_com_regularizacao(x, erro, M, sigma=1/2, reg = 0):
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
    
    print('\nIterações:', k)

    return x
