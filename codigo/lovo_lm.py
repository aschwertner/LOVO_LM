#******************************* PACOTES ***********************************
import numpy as np
import scipy.linalg as splinalg
import math as m

import funcoesauxiliares as fa

#****************************** PRINCIPAL **********************************

def lovolm(modelo, deparcial, damping, x, p, e, kmax, alfa, r, dados, verbose = False):
    """Objetivo: Determinar o melhor ajuste para o modelo, via Método de
    Levenberg-Marquardt com Busca Linear, aplicado ao problema LOVO.

    Entrada:
    modelo -  modelo a ser ajustado.
    deparcial - derivadas parciais do modelo.
    damping - parâmetro de damping
    x - vetor com os parâmetros iniciais.
    p - número de observações a serem consideradas.
    e - tolerância máxima.
    kmax - número máximo de iterações.
    alfa - parâmetro do método de busca linear.
    r - número total de observações.
    dados - endereço dos dados.
    verbose - Imprime informações acerca dos iterandos.
    Saída:
    Retorna os parâmetros do modelo após o ajuste e o número de iterações.
    """
    t_dados = []
    y_dados = []
    arq = open(dados, 'r')
    for i in range(r):
        aux = arq.readline()
        aux = aux.split()
        t_dados.append(float(aux[0]))
        y_dados.append(float(aux[1]))
    arq.close()
    t_dados = np.array(t_dados)
    y_dados = np.array(y_dados)

    res_mod = np.empty(r)
    jacobiana_res = np.empty([p, len(x)])
    quadres_mod = np.empty(r)
    d_aux = np.empty(len(x))

    k = 0
    ident = np.identity(len(x))
    dados_lovo = fa.riji(modelo, deparcial, x, t_dados, y_dados, p, r, res_mod, jacobiana_res)
    norma_grad = np.linalg.norm(dados_lovo[0], 2)
    x_aux = np.empty(len(x))
    dados_aux = 0.0
    dados_auxiliar = 0.0
    d_k = np.empty(len(x))

    if verbose:
        print('\n')
        print('--------------------------- DADOS ----------------------------')
            
    while norma_grad > e:
        dados_auxiliar = (1 / 2) * dados_lovo[1]
        if verbose:
            print('Iteração: ', k + 1)
        lambda_k = damping(dados_auxiliar, dados_lovo[0], k, jacobiana_res, d_aux)
        d_k = splinalg.cho_solve(splinalg.cho_factor(splinalg.blas.dgemm(1.0, jacobiana_res.T, jacobiana_res, lambda_k, ident), overwrite_a = True), -dados_lovo[0].T, overwrite_b = True)
        if verbose:
            print('Parâmetro de damping: ', lambda_k)
            print('Direção: ', d_k)
        l = 1.0
        x_aux[:] = x
        dados_aux = alfa * splinalg.blas.ddot(dados_lovo[0].T, d_k.T)
        if verbose:
            print('alfa * gradiente.T * d_k: ', dados_aux)
        while fa.funcobjetivo(modelo, splinalg.blas.daxpy(d_k, x_aux, len(d_k), l), t_dados, y_dados, p, r, quadres_mod) > dados_auxiliar + l * dados_aux:
            l = l / 2.0
            x_aux[:] = x
        x[:] = x_aux
        if verbose:
            print('Passo l: ', l)
            print('Iterando: ', x)
        k = k + 1
        if k > kmax:
            print('----------------------- ATENÇÃO! ------------------------')
            print('--------- Excedeu o número máximo de iterações! ---------')
            break
        dados_lovo = fa.riji(modelo, deparcial, x, t_dados, y_dados, p, r, res_mod, jacobiana_res)
        norma_grad = np.linalg.norm(dados_lovo[0], 2)
        if verbose:
            print('Norma do gradiente: ', norma_grad)
            print('Valor da função: ', (1 / 2) * dados_lovo[1])
            print('---------------------------------------------------------')
    return x, k, norma_grad
