# Método de Levenberg-Marquardt para Problemas de Otimização de Menor Valor Ordenado

## Introdução

Este repositório contêm implementações em *Python 3.6.7* do método LOVO-LM, descrito em

> Schwertner, A. E. (2019). O Método de Levenberg-Marquardt para Problemas de Otimização de Menor Valor Ordenado (Dissertação de Mestrado). Universidade Estadual de Maringá - UEM, Maringá, PR, Brasil.

e do método de Gaus-Newton para problemas LOVO, tratado em

> Andreani, R., Cesar, G., Cesar-Jr, R. M., Martínez, J. M., e Silva, P. J. S. (2007). Efficient curve detection using a Gauss-Newton method with applications in agriculture. In Proc. 1st International Workshop on Computer Vision Applications for Developing Regions in Conjunction with ICCV, 2007-CVDR-ICCV07.

Assim como arquivos relativos aos testes descritos na dissertação apontada acima.

## Como utilizar

 - Caso queira executar novamente os testes, basta alterar os campos *OPÇÕES* e *PARÂMETROS* presentes no arquivo *teste_algoritmos.py*, o qual se encontra na pasta *codigos*.

 - Caso queira executar algum algoritmo isoladamente, basta compilar seu respectivo código e chamar o método como uma função, atribuindo os parâmetros desejados. Mais informações estão disponíveis no próprio código ou utilizar o comando *help('nome_da_função')* depois de compilar.

## Testes

Os seguintes parâmetros de *damping* estão implementados:
![](/imagens/damping.png)

As funções modelo tomadas nos testes foram:
![](/imagens/modelos.png)







