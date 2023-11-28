import numpy as np

def quadratica(x, g=False):
    n = x.shape[0]
    vetor_i = np.arange(1, n+1)
    
    if g:
        return vetor_i * x
    
    return 1/2 * np.sum(vetor_i * x**2)

def rosenbrook(x, g=False):
    x_impares = x[::2]
    x_pares = x[1::2]
    
    if g:
        grad = np.zeros_like(x)
        # Gradientes dos elementos de índices ímpares
        grad[::2] = 20 * (x_pares - x_impares**2) * (- 2 * x_impares) + 2 * (x_impares - np.ones_like(x_impares))
        # Gradientes dos elementos de índices pares
        grad[1::2] = 20 * (x_pares - x_impares**2)
        return grad
    
    return np.sum(10*(x_pares - x_impares**2)**2 + (x_impares - np.ones_like(x_impares))**2)
    