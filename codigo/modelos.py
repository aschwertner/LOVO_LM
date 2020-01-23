#******************************* PACOTES ***********************************
import numpy as np
import math as m

#************************* FUNÇÕES ******************************************

def modelo01(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*t+x2.
    """
    return x[0] * t + x[1]


def derivadaparcial01(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([t, 1])

def modelo02(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*t^3+x2*t^2+x[3]*t+x[4].
    """
    return x[0] * t ** 3 + x[1] * t ** 2 + x[2] * t + x[3]


def derivadaparcial02(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([t ** 3, t ** 2, t, 1])

def modelo03(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*exp(x2*t)+x3.
    """
    return x[0] * m.exp(x[1] * t + x[2]) + x[3]


def derivadaparcial03(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([m.exp(x[1] * t + x[2]), x[0] * t * m.exp(x[1] * t + x[2]), x[0] * m.exp(x[1] * t + x[2]), 1])

def modelo04(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*sin(x2*t+x3)+x4.
    """
    return x[0] * m.sin(x[1] * t + x[2]) + x[3]


def derivadaparcial04(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([m.sin(x[1] * t + x[2]), x[0] * t * m.cos(x[1] * t + x[2]), x[0] * m.cos(x[1] * t + x[2]), 1])

def modelo05(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1*sin(x2*t)+x3*cos(x4*t)+x5.
    """
    return x[0] * m.sin(x[1] * t) + x[2] * m.cos(x[3] * t) + x[4]


def derivadaparcial05(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([m.sin(x[1] * t), t * x[0] * m.cos(x[1] * t), m.cos(x[3] * t), - t * x[2] * m.sin(x[3] * t), 1])

def modelo06(x,t):
    """Objetivo: Fornecer o modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna o modelo, com os parâmetros considerados. Ex: f(t)= x1/(1+exp(x2*t+x3)).
    """
    return x[0] / (1 + m.exp(x[1] * t + x[2]))


def derivadaparcial06(x,t):
    """Objetivo: Fornecer as derivadas parciais do modelo a ser considerado no ajuste.

    Entrada:
    x - vetor contendo os parâmetros a serem ajustados.
    t - variável.
    Saída:
    Retorna um vetor 1x2, com as derivadas parciais do modelo.
    """
    return np.array([1 / (1 + m.exp(x[1] * t + x[2])), - t * x[0] * m.exp(x[1] * t + x[2]) / ((1 + m.exp(x[1] * t + x[2])) ** 2),
                     - x[0] * m.exp(x[1] * t + x[2]) / ((1 + m.exp(x[1] * t + x[2])) ** 2)])
