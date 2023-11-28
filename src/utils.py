import numpy as np

def exportarDados(nome_arquivo):
    u = []
    v = []
    w = []
    
    with open(nome_arquivo, 'r') as arquivo:
        # Ler cada linha do arquivo
        linhas = arquivo.readlines()
        
        for linha in linhas:
            # Dividir a linha em varias partes usando espaço como separador
            partes = linha.strip().split()
            
            if len(partes) == 3:
                # Converter as partes para números de ponto flutuante
                num1 = float(partes[0])
                num2 = float(partes[1])
                num3 = float(partes[2])
                
                # Adicionar o par de números como uma tupla à lista
                u.append(num1)
                v.append(num2)
                w.append(num3)

            else:
                raise Exception("Arquivo deve ter 3 números por linha")

    return np.array(u), np.array(v), np.array(w)

def norma_inf(vetor):
    return np.max(np.abs(vetor))