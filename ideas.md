# Ideias para Analise


#### OBS: Essa divisão por sentença deles é meio ruim

- Eu vou conseguir varias datas, pegar a do IMDB e concatenar com a data normalizada. Eg: Homem aranha é de 2003, vou por a data como 2003 2000
- Verificar classificação indicativa e talvez inferir (???) eu posso pegar um monte de filme do imdb pegar o par (plot-classificação indicativa) e tentar fazer um classificador pra classificação indicativa? 
- Verificar genero e talvez inferir (???) eu posso pegar um monte de filme do imdb pegar o par (plot-genero) e tentar fazer um classificador pro genero? 
- Verificar media (Netflix, streaming, tv...)
- verificar linguagem
- Verificar premios
- Verificar nomes retirar os q forem marcados com comparação??



##  lendo consultas 0 eu achei isso:


- 0 -> dead dudes in the house does not have any plot
- 1 -> grown up 2 the plot is very large, also tthere is mentions of the rating and picture of the poster
- 2 -> date mention, also plot is very large
- 3 -> very specific description of a single scene
- 4 -> ???
- 5 -> simplesmente a descrição mais bizarra de v for vendetta
- 6 -> description is not that bad, can use date
- 7 -> tv show, so plots are too vague, every episode is a diferent thing (IMPOSSIBLE?)
- 8 -> there is an actual name here actor name, george clooney!
- 9 -> 
- 10 -> the plot is huge, a lot of diferent stories, the query only describes one (IMPOSSIBLE?)

Eu podia pegar um plot do IMDB, tem arquivo da wikipedia q nem tem. (e ai eu não acertei)


Modelos pesados e mirabolantes são dificeis, meu plano é tirar leite de pedra

heuristica pra dar match na data cm a q eu tenho, genero com o genero, meio com media (streaming, cinema...), os nomes disponiveis.