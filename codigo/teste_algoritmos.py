#******************************** PACOTES *************************************

from numpy import array, empty
from timeit import default_timer
from random import uniform, seed
from sys import exc_info

import funcoesauxiliares as fa
import modelos as mods
import lovo_lm as lm
import lovo_gn as gn

#******************************** OPÇÕES **************************************

verbose_1 = False          #Executa o Algoritmo LOVO Gauss-Newton
verbose_2 = False          #Executa o Algoritmo LOVO Levenber-Marquardt
verbose_3 = True         #Executa o Algoritmo LOVO Levenberg-Marquardt apenas
                              #para os parâmetros de damping escolhido em "pd".
verbose_4 = True         #Imprime informações sobre as iterações

#******************************* PARÂMETROS ***********************************

modelo = 2                #Escolha de 0 a 5:
                                          #0:= polinômio de grau 01 (2 param.)
                                          #1:= polinômio de grau 03 (4 param.)
                                          #2:= exponencial (4 param.)
                                          #3:= senóide 01 (4 param.)
                                          #4:= senóide 02 (5 param.)
                                          #5:= logística (3 param.)
num_pontos = 1            #Escolha de 0 a 2:
                                          #0:= 100 pontos (90% conf.)
                                          #1:= 1000 pontos (90% conf.)
                                          #2:= 5000 pontos (90% conf.)
pd = 7                    #Caso 'verbose_3 = True', escolha de 0 a 7:
                                          #0:= PD1 (Levenberg)
                                          #1:= PD2 (Levenberg mod. 01)
                                          #2:= PD3 (Levenberg mod. 02)
                                          #3:= PD4 (Yamashita e Fukushima)
                                          #4:= PD5 (Fan e Yuan)
                                          #5:= PD6 (Benatti)
                                          #6:= PD7 (Schwertner 01)
                                          #7:= PD7 lambda_min = 10**(-2)
                                                  #(Schwertner 01, híbrido 01)
                                          #8:= PD7 lambda_min = 1
                                                  #(Schwertner 01, híbrido 02)
                                          #9:= PD7 (Schwertner 01, híbrido 03)
                                          #10:= PD8 lambda_min = 10
                                                  #(Schwertner 02)
                                          #11:= PD8 lambda_min = 10**(-2)
                                                   #(Schwertner 02, híbrido 01)
                                          #12:= PD8 lambda_min = 1
                                                   #(Schwertner 02, híbrido 02)
                                          #13:= PD8 lambda_min = 10
                                                   #(Schwertner 02, híbrido 03)
tolerancia = 10 ** (-4)   #Tolerância a ser observada
max_iteracoes = 500       #Número máximo de iterações
alfa = 10 ** (-4)         #Parâmetro da Busca Linear
beta = 10 ** (-2)         #Parâmetro de correção para o Alg. Lovo Gauss-Newton

#********************************* DADOS **************************************

dados = ['../dados/01_dados_100pt.dat',
         '../dados/01_dados_1000pt.dat',
         '../dados/01_dados_5000pt.dat',
         '../dados/02_dados_100pt.dat',
         '../dados/02_dados_1000pt.dat',
         '../dados/02_dados_5000pt.dat',
         '../dados/03_dados_100pt.dat',
         '../dados/03_dados_1000pt.dat',
         '../dados/03_dados_5000pt.dat',
         '../dados/04_dados_100pt.dat',
         '../dados/04_dados_1000pt.dat',
         '../dados/04_dados_5000pt.dat',
         '../dados/05_dados_100pt.dat',
         '../dados/05_dados_1000pt.dat',
         '../dados/05_dados_5000pt.dat',
         '../dados/06_dados_100pt.dat',
         '../dados/06_dados_1000pt.dat',
         '../dados/06_dados_5000pt.dat']
modelos = [mods.modelo01, mods.modelo02, mods.modelo03, mods.modelo04,
           mods.modelo05, mods.modelo06]
derivadas = [mods.derivadaparcial01, mods.derivadaparcial02,
             mods.derivadaparcial03, mods.derivadaparcial04,
             mods.derivadaparcial05, mods.derivadaparcial06]
def_modelos = ['Função Polinomial de grau 1', 'Função Polinomial de grau 3',
               'Função Exponencial', 'Função Senoidal 01', 'Função Senoidal 02',
               'Função Logística']
pontos_confiaveis = [90, 900, 1000]
pontos_total = [100, 1000, 5000]
damping = [fa.pd1, fa.pd2, fa.pd3, fa.pd4, fa.pd5, fa.pd6, fa.pd7l0, fa.pd7l1,
           fa.pd7l2, fa.pd7l3, fa.pd8l0, fa.pd8l1, fa.pd8l2, fa.pd8l3]
damping_nomes = ['damping Levenberg', 'damping Levenberg Mod.01',
                 'damping Levenberg Mod.02', 'damping Yamashita e Fukushima', 
                 'damping Fan e Yuan', 'damping Benatti',
                 'damping Schwertner 01',
                 'damping Schwertner 01, com lambda_min = 10**(-2)',
                 'damping Schwertner 01, com lambda_min = 1',
                 'damping Schwertner 01, com lambda_min = 10',
                 'damping Schwertner 02',
                 'damping Schwertner 02, com lambda_min = 10**(-2)',
                 'damping Schwertner 02, com lambda_min = 1',
                 'damping Schwertner 02, com lambda_min = 10']
x0 = [array([0.0, 0.0]), array([0.0, 0.0, 0.0, 0.0]), array([0.0, 0.0, 0.0, 0.0]),
      array([1.0, 1.0, 1.0, 1.0]), array([5.0, 5.0, 5.0, 5.0, 5.0]),
      array([1.0, 0.0, 0.0])] #Parâmetros iniciais.

solucoes = ['[-3.2531, 15.2347]',
            '[1.12548, 2.53168, 3.14724, 0.589134]',
            '[5.3127, -0.252, -0.755, -25.64]',
            '[40.5367, 2.345, -5.234, 24.12]',
            '[13.535, -11.86, 12.8239, -6.30077, 4.82982]',
            '[-10.5772, -4.52081, 19.6434]']

#******************************* PRINCIPAL ************************************

print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print('*                                TESTE                                *')
print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print('Problema: ', def_modelos[modelo])
print('Número de pontos: ', pontos_total[num_pontos])
print('Solução exata: ', solucoes[modelo])
print('Parâmetros iniciais: ', x0[modelo], '\n')
print('*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* \n \n')

d = 3 * modelo + num_pontos
x_0 = empty(len(x0[modelo]))

if verbose_1:
    x_0[:] = x0[modelo]
    try:
        print('------------------- LOVO Gauss-Newton -------------------')
        inicio = default_timer()
        print(gn.lovogn(modelos[modelo], derivadas[modelo], x_0,
                        pontos_confiaveis[num_pontos], tolerancia, max_iteracoes,
                        alfa, beta, pontos_total[num_pontos], dados[d],
                        verbose_4))
        fim = default_timer()
        print('A execução levou ', fim - inicio, 'segundos. \n \n')
    except OverflowError as overf:
        print('----------------------- ATENÇÃO! ------------------------')
        print('----- Houve um erro durante a execução do algoritmo -----')
        print('--- Erro de Overflow: ', overf, ' ---')
        print('--------------------------------------------------------- \n \n')
    except:
        print('----------------------- ATENÇÃO! ------------------------')
        print('----- Houve um erro durante a execução do algoritmo -----')
        print('Erro inesperado: ', exc_info()[0:2], ' ---')
        print('--------------------------------------------------------- \n \n')

if verbose_2:
    for i in range(len(damping)):
        try:
            print('--- LOVO Levenberg-Marquardt (', damping_nomes[i],') ---')
            x_0[:] = x0[modelo]
            inicio = default_timer()
            print(lm.lovolm(modelos[modelo], derivadas[modelo], damping[i], x_0,
                            pontos_confiaveis[num_pontos], tolerancia,
                            max_iteracoes, alfa, pontos_total[num_pontos], dados[d],
                            verbose_4))
            fim = default_timer()
            print('A execução levou ', fim - inicio, 'segundos. \n \n')
        except OverflowError as overf:
            print('----------------------- ATENÇÃO! ------------------------')
            print('----- Houve um erro durante a execução do algoritmo -----')
            print('--- Erro de Overflow: ', overf, ' ---')
            print('--------------------------------------------------------- \n \n')
        except:
            print('----------------------- ATENÇÃO! ------------------------')
            print('----- Houve um erro durante a execução do algoritmo -----')
            print('Erro inesperado: ', exc_info()[0:2], ' ---')
            print('--------------------------------------------------------- \n \n')         

if verbose_3:
    try:
        print('--- LOVO Levenberg-Marquardt (', damping_nomes[pd],') ---')
        x_0[:] = x0[modelo]
        inicio = default_timer()
        print(lm.lovolm(modelos[modelo], derivadas[modelo], damping[pd], x_0,
                            pontos_confiaveis[num_pontos], tolerancia,
                            max_iteracoes, alfa, pontos_total[num_pontos], dados[d],
                            verbose_4))
        fim = default_timer()
        print('LOVO Levenberg-Marquardt (', damping_nomes[pd],'): '
                  'A execução levou ', fim - inicio, 'segundos. \n \n')
    except OverflowError as overf:
        print('----------------------- ATENÇÃO! ------------------------')
        print('----- Houve um erro durante a execução do algoritmo -----')
        print('--- Erro de Overflow: ', overf, ' ---')
        print('--------------------------------------------------------- \n \n')
    except:
        print('----------------------- ATENÇÃO! ------------------------')
        print('----- Houve um erro durante a execução do algoritmo -----')
        print('Erro inesperado: ', exc_info()[0:2], ' ---')
        print('--------------------------------------------------------- \n \n')
