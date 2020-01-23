#******************************* PACOTES ***********************************
import numpy as np
import math as m

import scipy.linalg as splinalg

#************************* FUNÇÕES AUXILIARES ******************************

def riji(modelo, deparcial, x, t, y, p, r, residuos, jacobiana):
    """Objetivo: Fornecer informações gerais relacionadas aos p menores
    resíduos.

    Entrada:
    modelo - modelo a ser ajustado.
    deparcial - derivadas parciais do modelo.
    x - vetor contendo os parâmetros a serem ajustados.
    t - vetor contendo as variáveis tomadas nas observações.
    y - vetor contendo os resultados obtidos nas observações.
    p - número de observações a serem consideradas.
    r - número total de observações.
    residuos - vetor pré-inicializado.
    jacobiana - matriz pré-inicializada.
    Saída:
    Retorna uma tupla com o gradiente dos p menores resíduos e
    a soma dos quadrados dos p menores resíduos.
    Altera as entradas do vetor "residuos", atribuindo as primeiras p
    entradas aos p menores resíduos.
    Altera a matriz "jacobiana", com o valor da jacobiana calculada nos p
    menores resíduos.
    """    
    
    for i in range(r):
        residuos[i] = modelo(x, t[i]) - y[i]

    ind_mod = np.arange(0, r)
    for i in range(p):
        p_min = i
        for j in range(p_min + 1, r):
            if residuos[j] ** 2 < residuos[p_min] ** 2:
                p_min = j
        temp_1 = residuos[i]
        temp_2 = ind_mod[i]
        residuos[i] = residuos[p_min]
        ind_mod[i] = ind_mod[p_min]
        residuos[p_min] = temp_1
        ind_mod[p_min] = temp_2
        
    soma_quadres = 0
    for i in range(p):
        soma_quadres = soma_quadres + residuos[i] ** 2
    
    for j in range(p):
        jacobiana[j, :] = deparcial(x, t[ind_mod[j]])
    
    return splinalg.blas.dgemv(1.0, jacobiana.T, residuos[0 : p].T), soma_quadres

def funcobjetivo(modelo, x, t, y, p, r, quadresiduos):
    """Objetivo: Calcula o valor da função objetivo.

    Entrada:
    modelo - modelo a ser ajustado.
    x - vetor contendo os parâmetros a serem ajustados.
    t - vetor contendo as variáveis tomadas nas observações.
    y - vetor contendo os resultados obtidos nas observações.
    p - número de observações a serem consideradas.
    r - número total de observações.
    quadresiduos - vetor pré-inicializado.
    Saída:
    Retorna metade do valor da soma dos quadrados dos p menores resíduos.
    """
    for i in range(r):
        quadresiduos[i] = (modelo(x, t[i]) - y[i]) ** 2

    soma_quadres = 0
    for i in range(p):
        p_min = i
        for j in range(p_min + 1, r):
            if quadresiduos[j] < quadresiduos[p_min]:
                p_min = j
        temp_1 = quadresiduos[i]
        quadresiduos[i] = quadresiduos[p_min]
        quadresiduos[p_min] = temp_1
        soma_quadres = soma_quadres + quadresiduos[i]

    return (1 / 2) * soma_quadres

def pd1(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Levenberg.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return (np.linalg.norm(gradfuncao, 2)) ** 2 / funcao

def pd2(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Levenberg Modificado 01.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return (np.linalg.norm(gradfuncao, 2)) ** 2

def pd3(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Levenberg Modificado 02.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return (np.linalg.norm(gradfuncao, 2))

def pd4(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Yamashita e Fukushima.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return 2 * funcao

def pd5(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Fan e Yuan.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return (2 * funcao) ** (1 / 2)

def pd6(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Benatti.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return 2 * np.linalg.norm(gradfuncao, 2) / (3 * (iteracao + 1))

def pd7l0(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 01.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return 2 * funcao ** (1 / 2) / (3 * (iteracao + 1))

def pd7l1(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 01,
    híbrido 01 (lambda_min = 10**(-2)).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao ** (1 / 2) / (3 * (iteracao + 1)), 10.0 ** (-2))

def pd7l2(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 01,
    híbrido 02 (lambda_min = 1).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao ** (1 / 2) / (3 * (iteracao + 1)), 1.0)

def pd7l3(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 01,
    híbrido 03 (lambda_min = 10).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao ** (1 / 2) / (3 * (iteracao + 1)), 10.0)

def pd8l0(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 02.

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return 2 * funcao / (3 * (iteracao + 1))

def pd8l1(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 02,
    híbrido 01 (lambda_min = 10**(-2)).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao / (3 * (iteracao + 1)), 10.0 ** (-2))

def pd8l2(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 02,
    híbrido 02 (lambda_min = 1).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao / (3 * (iteracao + 1)), 1.0)

def pd8l3(funcao, gradfuncao, iteracao, jacobiana, d):
    """Objetivo: Calcular o parâmetro de 'damping', por Schwertner 02,
    híbrido 03 (lambda_min = 10).

    Entrada:
    funcao - valor da função objetivo.
    gradfuncao - valor do gradiente da função objetivo.
    Saída:
    Retorna o parâmetro de 'damping'.
    """
    return max(2 * funcao / (3 * (iteracao + 1)), 10.0)
