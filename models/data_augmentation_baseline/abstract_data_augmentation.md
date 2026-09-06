# Data Augmentation em três abordagens para ARC-AGI

## 1. Visão geral

Os três artigos exploram o mesmo problema: em ARC-AGI, o modelo recebe poucos pares de entrada e saída e precisa inferir a regra que os relaciona para produzir uma nova saída. Como há poucos exemplos, uma única leitura da tarefa pode induzir o modelo a um erro de perspectiva, de ordem dos exemplos ou de interpretação da regra.

A ideia comum é criar várias versões da mesma tarefa e usar essas versões de maneira consistente. Entretanto, os objetivos são diferentes:

- **The LLM ARChitect** usa augmentation como um mecanismo de perspectiva, geração de candidatos e seleção probabilística.
- **TTT for Abstract Reasoning**, do MIT, usa augmentation primeiro para treinar um adaptador específico da tarefa no test-time e depois para gerar previsões que serão agregadas por votação.
- **BARC** combina um ramo indutivo, que sintetiza programas Python verificáveis, com um ramo transdutivo baseado em TTT e augmentation.

Assim, augmentation não significa apenas aumentar um dataset de treinamento. Nos três trabalhos, ele é usado principalmente durante a inferência para produzir evidência adicional sobre qual solução é consistente.

## 2. The LLM ARChitect: resolver por perspectivas

### Objetivo

O artigo parte da observação de que um LLM lê uma matriz como uma sequência unidimensional de tokens. Uma regra espacial pode ser difícil na orientação original, mas mais simples quando a matriz é rotacionada, refletida ou transposta. A aumentação muda a apresentação da tarefa sem mudar sua regra abstrata.

### Transformações

As principais transformações são:

- **Simetrias geométricas do grupo diédrico D8**: identidade, rotações de 90, 180 e 270 graus e reflexões horizontais, verticais e diagonais/transposição.
- **Permutações de cores**: troca bijetiva dos identificadores de cor, normalmente preservando o papel especial do fundo quando necessário.
- **Reordenação dos exemplos**: embaralhamento dos pares entrada-saída no prompt para reduzir dependência da posição de cada demonstração.

As transformações usadas para gerar respostas precisam ser reversíveis. Se o modelo produz uma resposta na visão transformada, o sistema aplica a transformação inversa para levá-la de volta ao espaço original.

### Geração de candidatos

Para cada tarefa são criadas aproximadamente 8 a 16 visões aumentadas. Em cada visão, o modelo usa uma busca em profundidade (DFS) guiada pela probabilidade dos tokens. A busca mantém sequências plausíveis e poda caminhos cuja probabilidade acumulada fica abaixo do limiar configurado.

Cada saída completa é convertida de volta para a orientação original. As respostas duplicadas são removidas, formando um conjunto global de candidatos. Essa etapa prioriza **recall**: a resposta correta deve aparecer entre os candidatos mesmo que não seja a mais provável na visão original.

### Seleção com AugScore

Depois da geração, cada candidato é avaliado em todas as simetrias D8. Para uma transformação $T_i$, calcula-se a probabilidade da resposta transformada condicionada à tarefa também transformada:

$$
P_i(S) = P_M\big(T_i(S) \mid T_i(C)\big)
$$

O score é a soma dos logaritmos dessas probabilidades:

$$
\operatorname{AugScore}(S) = \sum_i \log P_M\big(T_i(S) \mid T_i(C)\big)
$$

Isso equivale a maximizar o produto das probabilidades. Uma resposta correta tende a continuar coerente em várias perspectivas; uma alucinação pode parecer boa em uma orientação e tornar-se improvável em outra. A soma de log-probabilidades penaliza fortemente esse tipo de instabilidade.

### Característica principal

O LLM ARChitect não depende principalmente de atualizar os pesos do modelo. Sua força vem de explorar várias representações da mesma tarefa, gerar muitos candidatos e selecionar aqueles que apresentam maior consistência probabilística entre as visões.

## 3. MIT: Test-Time Training com dataset sintético

### Objetivo

O método do MIT usa os exemplos disponíveis na própria tarefa para adaptar temporariamente o modelo. Como a saída do exemplo de teste real é desconhecida, os exemplos de demonstração são reutilizados como pequenos problemas supervisionados.

### Leave-one-out

Para cada par conhecido $(x_j, y_j)$, o sistema remove esse par do contexto e o transforma em um exemplo de teste sintético. Os demais pares permanecem como demonstrações:

$$
\{(x_k,y_k): k \ne j\} \rightarrow (x_j,y_j)
$$

Dessa forma, o modelo treina para prever uma saída conhecida usando exatamente o tipo de contexto que encontrará na tarefa real. Também são usadas permutações da ordem dos exemplos para reduzir vieses de posição e recência.

### Transformações no dataset de TTT

As tarefas leave-one-out são aumentadas com transformações baseadas em regras, aplicadas de maneira consistente às entradas e saídas:

- rotações de 90, 180 e 270 graus;
- flips horizontal e vertical;
- transposição/reflexão diagonal;
- reflexões com concatenação, anexando a imagem espelhada à esquerda, direita, topo ou base;
- translações aleatórias nos eixos $X$ e $Y$, com deslocamento limitado;
- aumento de resolução, de altura ou de largura;
- composição de transformações, como rotação seguida de aumento de resolução;
- permutação de cores;
- embaralhamento da ordem dos exemplos.

O dataset sintético é limitado a cerca de 250 exemplos por tarefa. A limitação controla custo e evita que o ajuste específico domine o modelo.

### Adaptação via LoRA

O modelo treina adaptadores LoRA exclusivos para cada tarefa, mantendo os pesos principais congelados. O treinamento dura duas épocas e usa cross-entropy para ensinar o modelo a reproduzir as saídas dos exemplos sintéticos.

A augmentation tem aqui uma função mais profunda do que simplesmente fornecer opiniões diferentes: ela cria variações supervisionadas que obrigam o adaptador a capturar a regra, e não apenas memorizar a aparência ou a orientação de uma matriz.

A ablação relatada no artigo mostra a importância dessa etapa: remover as transformações do TTT reduz substancialmente o número de tarefas resolvidas, de 29 para 13 no experimento citado.

### Inferência aumentada

Após o ajuste do LoRA, a tarefa real é apresentada em várias transformações geométricas invertíveis. Para cada transformação e para duas ordens de demonstrações, o modelo produz uma resposta usando decodificação gulosa. As respostas são então transformadas de volta à orientação original.

### Votação hierárquica

As previsões não são simplesmente misturadas em uma votação plana:

1. As previsões são agrupadas pela transformação aplicada.
2. Em cada grupo, selecionam-se os três candidatos mais frequentes.
3. Se necessário, são criados candidatos auxiliares por maioria de linhas ou de colunas.
4. Os candidatos selecionados de todos os grupos participam de uma votação global.
5. Em empates, dá-se prioridade à previsão da transformação identidade.
6. As duas respostas mais votadas são mantidas para o formato pass@2 do ARC.

### Característica principal

O MIT combina **adaptação de parâmetros** e **ensemble de previsões**. O modelo aprende temporariamente a tarefa a partir de dados aumentados e depois usa várias perspectivas para estabilizar a resposta.

## 4. BARC: híbrido entre programas e transdução

### Objetivo

BARC combina dois paradigmas:

- **Indução**: o modelo gera programas Python que devem explicar todos os exemplos de treino.
- **Transdução**: o modelo gera diretamente a matriz de saída, sem escrever um programa explícito.

A augmentation aparece principalmente no ramo transdutivo, enquanto a verificação do ramo indutivo é feita pela execução dos programas nos exemplos conhecidos.

### Augmentation no TTT

Quando o ramo indutivo não encontra um programa válido, BARC constrói pseudo-tarefas com leave-one-out. Para cada exemplo conhecido, um par é tratado como teste falso e os demais são usados como contexto. Em seguida, são geradas aproximadamente dez variações aumentadas por pseudo-tarefa, usando:

- permutações de cores;
- rotações de 90, 180 e 270 graus;
- reflexões horizontais, verticais e diagonais.

Esse procedimento produz aproximadamente 12 mil instâncias derivadas das tarefas de validação. Para evitar esquecimento catastrófico e tornar o ajuste menos estreito, o dataset ainda mistura 5 mil exemplos aumentados de Re-ARC e 5 mil exemplos sintéticos de ARC-Heavy, totalizando cerca de 22 mil problemas.

### Adaptação no test-time

O BARC treina um adaptador LoRA no test-time por três épocas. O adaptador é usado apenas no ramo transdutivo e serve para ajustar o modelo à distribuição e às regularidades da tarefa atual. Depois do TTT, a resposta é produzida com beam search, com largura entre 20 e 40.

### Ramo indutivo: amostragem e verificação

Independentemente do TTT, o ramo indutivo amostra de 10 mil a 20 mil programas Python. Cada programa é executado em todos os exemplos de treino:

$$
 f(x_k)=y_k \quad \text{para todo } k
$$

Apenas os programas que passam nessa verificação determinística são considerados válidos. As previsões desses programas para a entrada de teste são agregadas por maioria. Como um programa válido precisa explicar os exemplos observados, essa etapa fornece uma forma de consistência simbólica que o ramo transdutivo não possui.

### Ensemble final

O sistema prefere a previsão do ramo indutivo quando existe pelo menos um programa válido. Se nenhum programa passa nos exemplos conhecidos, usa a previsão transdutiva produzida pelo modelo adaptado com TTT.

Em termos conceituais, a augmentation ajuda o ramo transdutivo a aprender e decodificar a tarefa; a síntese de programas fornece um mecanismo separado de verificação exata. O ganho final vem da combinação dos dois.

## 5. Comparação direta

| Dimensão | LLM ARChitect | MIT TTT | BARC |
|---|---|---|---|
| Papel principal da augmentation | Mudar a perspectiva e testar estabilidade | Criar dados supervisionados e gerar previsões variadas | Adaptar a transdução; complementar a indução simbólica |
| Adaptação dos pesos | Não é o foco principal | LoRA por tarefa, duas épocas | LoRA no test-time, três épocas |
| Construção de dados | Visões transformadas da tarefa | Leave-one-out + transformações + ordem embaralhada | Leave-one-out + transformações + Re-ARC e ARC-Heavy |
| Transformações | D8, cores e ordem dos exemplos | Geometria, cores, translação, resolução, concatenação e composição | Rotações, flips, diagonais e cores |
| Geração | DFS probabilística | Greedy | Program sampling no ramo indutivo e beam search no transdutivo |
| Seleção | Soma de log-probabilidades, AugScore | Votação hierárquica | Verificação de programas + majority vote; fallback transdutivo |
| Tipo de consistência | Probabilística entre perspectivas | Frequência entre grupos de transformações | Exata para programas e probabilística para transdução |
| Principal vantagem | Alto recall e bom filtro de candidatos | Adaptação forte com poucos exemplos | Combina flexibilidade neural e verificação simbólica |
| Principal custo | Muitas buscas e avaliações de candidatos | Treinamento por tarefa e muitas inferências | Amostragem de milhares de programas mais TTT |

## 6. Semelhanças

1. **Transformações são aplicadas de forma consistente**: quando uma tarefa é rotacionada ou refletida, entradas e saídas são transformadas juntas.
2. **Reversibilidade é essencial**: a resposta produzida numa visão aumentada precisa ser mapeada de volta para a tarefa original.
3. **O objetivo é reduzir dependência de uma única perspectiva**: uma solução que aparece apenas numa orientação é menos confiável.
4. **Há algum tipo de agregação**: AugScore no LLM ARChitect, votação hierárquica no MIT e verificação/votação no BARC.
5. **O custo é deslocado para o test-time**: os métodos usam computação adicional quando a tarefa já chegou, em vez de depender exclusivamente de treinamento prévio.

## 7. Diferenças fundamentais

A diferença mais importante está no que cada artigo considera evidência de que uma resposta está correta:

- Para o **LLM ARChitect**, é a alta probabilidade da mesma solução em várias perspectivas.
- Para o **MIT**, é a frequência da solução entre transformações e ordens de contexto, depois de um ajuste específico da tarefa.
- Para o **BARC**, no ramo indutivo, é a execução de um programa que reproduz exatamente todos os exemplos conhecidos; no ramo transdutivo, é a consistência obtida por TTT e beam search.

Também há uma diferença na função das transformações. No LLM ARChitect, elas são principalmente uma ferramenta de busca e reranking. No MIT, são parte do próprio conjunto de treinamento. No BARC, são usadas para produzir dados do TTT, mas o sistema ainda depende de um mecanismo simbólico paralelo.

## 8. Síntese final

Os três artigos tratam a aumentação como uma forma de **test-time compute orientado por invariâncias**. Em vez de aceitar a primeira saída do modelo, eles perguntam se a mesma regra continua produzindo uma resposta coerente quando a tarefa é vista de outro modo.

O LLM ARChitect explora melhor o espaço de candidatos e escolhe pela consistência probabilística. O MIT transforma os exemplos da tarefa em treinamento supervisionado temporário e combina as previsões com votação. BARC amplia essa ideia com um sistema híbrido: usa TTT e augmentation quando a transdução é necessária, mas tenta primeiro obter programas que possam ser verificados diretamente.

Portanto, as técnicas podem ser vistas como três níveis de compromisso:

1. **Múltiplas perspectivas**: gerar e ranquear respostas sem alterar significativamente o modelo.
2. **Adaptação temporária**: alterar um pequeno conjunto de parâmetros usando exemplos aumentados da própria tarefa.
3. **Verificação simbólica híbrida**: exigir que programas candidatos expliquem os exemplos e usar a transdução como fallback.

A conclusão comum é que, em ARC-AGI, a qualidade da inferência depende tanto da capacidade do modelo quanto da maneira como o problema é apresentado, repetido, transformado e validado durante o test-time.
