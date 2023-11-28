import numpy as np
import matplotlib.pyplot as plt
from time import time_ns
from src.utils import exportarDados
    
# Dados de treino
u, v = exportarDados('dados/QM01')

# Encontrar o vetor x = [x1, x2, x3] que minimiza a função perda
x_inicial = np.array([0,0,0])
erro = 1e-5
M = 10000

inicio = time_ns()
x_otimo = minimizador_nao_monotono(x_inicial, erro, M, sigma=0.3)
fim = time_ns()

tempo_execucao = (fim - inicio) * 1e-9
print('\nTempo de execução %.4f' % tempo_execucao)
print('\nX* =', x_otimo, '\n')

# Data for plotting
plt.title('Problema de Regressão')
plt.scatter(u,v)
v_pred = x_otimo[0] + x_otimo[1]*np.exp(x_otimo[2]*u)
plt.plot(u, v_pred, '-r')
plt.show()