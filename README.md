# SVM-TCC
Trabalho de conclusão de curso: APLICAÇÃO DE MÁQUINAS DE VETORES DE SUPORTE PARA GERAÇÃO DE SINAIS DE CONTROLE ROBUSTOS PARA PRÓTESES MIOELÉTRICAS

**Código feito em Python 3.6**

## DataSet file
Essa pasta contém as coletas de sinais EMG de 6 voluntários utilizando um bracelete com quatro canais EMG e realizando 7 tipos de movimentos diferentes. Cada movimento possui 160 amostras. O protocólo de coleta pode ser encontrado na minha monografia clicando [aqui](https://repositorio.ufu.br/handle/123456789/22274). Minha monografia explica de forma detalhada o passo a passo do experimento. Além disso, existem tópicos para um breve resumo sobre as principais técnicas utilizadas: SVM, PCA, Eletromiografia, etc.

## O que o código faz?

* **Processamento de dados**  
A primeira parte do [código principal](https://github.com/eberlawrence/SVM-TCC/blob/master/TrabalhoFinal.py) realiza a leitura e em sequência o processamento dos dados disponibilizados na pasta [DataSet](https://github.com/eberlawrence/SVM-TCC/tree/master/DataSet).

* **Extração de atributos**  
A segunda parte consiste na extração de atributos dos sinais EMG no domínio do tempo. Os atributos foram:
	* *Mean Absolute Value - MAV*
	* *Root Mean Square - RMS*
	* *ZC - Zero-Crossings*
	* *Signal Variance - VAR*
	* *Slope Sign Change - SSC*
	* *Waveform Length - WL*

* **Treinamento - Gerando um modelo**  
A terceira parte foi o treino do classificador para a geração de um modelo. O modelo foi criado utilizando a Técnica de *Support Vector Machine - SVM*.

* **Teste - Vizualizando os Resultados**  
Na quarta parte foram realizados testes do classificador.

* **Redução de dimensionalidade**  
Por último é feita a redução de dimensionalidade dos atributos extraídos usando a técnica de *Principal Component Analysis - PCA*. A figura abaixo mostra um exemplo, conseguindo reduzir a dimensionalidade de um espaço de 24D para 3D com 87% da variabilidade dos dados.

![PCA reduction](https://github.com/eberlawrence/SVM-TCC/blob/master/DataSet/SVM-PCA.JPG)


**Autor**: Eber Lawrence Souza Gouveia  
[Website](https://eberlawrence.github.io/) - [Linkedin](https://www.linkedin.com/in/ebersoulz/)  
[Laboratório de Engenharia Biomédica - BioLab](http://www.biolab.eletrica.ufu.br/)



