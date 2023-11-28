import numpy as np
from funcoes_especiais import quadratica, rosenbrook

#def fun(x, g=False):
#    if g:
#        v = np.zeros_like(x)
#        v[0] = - np.cos(x[0])
#        v[-1] = 3
#        return 2*x + v
#    
#    return np.sum(x**2) - np.sin(x[0]) + 3*x[-1]

fun = rosenbrook

def minimizador(x, erro, M):
    k = 0
    g_k = fun(x, g=True)
    
    while np.max(np.abs(g_k)) >= erro and k < M:
        d = - g_k
        t = 1
        while fun(x+t*d) >= fun(x):
            t = t/2 
        x = x + t*d
        g_k = fun(x, g=True)
        k = k + 1
    
    print('Iterações:',k )
    
    return x

def minimizador_armijo(x, erro, M, alpha=10e-4, sigma=1/2):
    k = 0
    g_k = fun(x, g=True)
    
    while np.max(np.abs(g_k)) >= erro and k < M:
        d = - g_k
        t = 1
        
        while fun(x+t*d) > fun(x) + alpha * t * np.dot(g_k, d):
            t = sigma*t
        x = x + t*d
        g_k = fun(x, g=True)
        k = k + 1
    
    print('Iterações:', k)
    
    return x

def minimizador_armijo_modificado(x0, erro, M, alpha=10e-4, sigma=1/2):
    k = 0
    x = [x0]
    g_k = [fun(x0, g=True)]
    
    while np.max(np.abs(g_k[-1])) >= erro and k < M:
        d = - g_k[-1]
        
        if  k == 0:
            t = 1
        else:
            t = np.linalg.norm(x[-1] - x[-2]) / np.linalg.norm(g_k[-1] - g_k[-2])
            
        while fun(x[-1] + t * d) > fun(x[-1]) + alpha * t * np.dot(g_k[-1], d):
            t = sigma*t
        
        x.append(x[-1] + t*d)
        g_k.append(fun(x[-1], g=True))
        k = k + 1
    
    print('Iterações:', k)
    
    return x[-1]

# Alterar
x0 = np.array([-1,3,-5,-9])
erro = 1/1000
M = 1000

#x_min = minimizador(x0, erro,M)
#
#print('Minimizador (sem Armijo):', x_min)
#print('Mínimo:', fun(x_min))
#print('')

x_min1 = minimizador_armijo(x0, erro, M, alpha=0.5)

print('Minimizador (com Armijo):', x_min1)
print('Mínimo:', fun(x_min1))

x_min2 = minimizador_armijo_modificado(x0, erro, M, alpha=0.5)

print('Minimizador (com Armijo modificado):', x_min2)
print('Mínimo:', fun(x_min2))
