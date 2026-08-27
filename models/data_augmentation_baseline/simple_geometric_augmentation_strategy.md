# Estratégia Simples de Data Augmentation na Inferência

Este documento descreve uma estratégia de baixo custo, inspirada principalmente no artigo *The Surprising Effectiveness of Test-Time Training for Abstract Reasoning* (MIT), mas sem implementar todo o pipeline de Test-Time Training (TTT).

A proposta combina:

1. Transformações geométricas simples da tarefa;
2. Uma predição gulosa do modelo para cada transformação;
3. Conversão de cada resposta de volta para a orientação original;
4. Votação global entre as respostas normalizadas.

O objetivo não é reproduzir a estratégia mais poderosa dos artigos, mas criar uma primeira versão fácil de implementar, testar e comparar com uma linha de base.

---

## 1. Ideia central

Um modelo de linguagem normalmente recebe cada grid serializado em uma ordem fixa, por exemplo, linha por linha. Essa ordem pode favorecer algumas regras espaciais e dificultar outras.

A mesma tarefa pode ficar mais fácil se for apresentada, por exemplo, transposta ou espelhada. Como essas transformações preservam a estrutura do problema, a resposta produzida na visão transformada pode ser convertida de volta para a orientação original.

A estratégia é:

```text
Tarefa original
    |
    +--> identidade       --> modelo --> resposta 1
    +--> flip horizontal  --> modelo --> resposta 2
    +--> flip vertical    --> modelo --> resposta 3
    +--> transposição     --> modelo --> resposta 4
                                             |
                         desfazer transformações
                                             |
                               votação global
                                             |
                                  resposta final
```

A resposta final não é escolhida pela probabilidade interna do modelo. Ela é escolhida pela frequência com que cada grid normalizado aparece entre as predições.

---

## 2. O que é transformado?

A transformação deve ser aplicada de maneira consistente à tarefa inteira:

- todos os pares de treinamento `(input, output)`;
- a entrada de teste;
- a saída produzida pelo modelo, quando ela for convertida de volta.

Se a tarefa original for:

```text
(input_1, output_1),
(input_2, output_2),
input_test
```

uma transformação `T` produz:

```text
(T(input_1), T(output_1)),
(T(input_2), T(output_2)),
T(input_test)
```

O modelo recebe essa tarefa transformada e gera uma saída `y_aug`. Em seguida, aplica-se a transformação inversa:

$$
\hat{y} = T^{-1}(y_{aug})
$$

Assim, todas as respostas ficam representadas na mesma orientação e podem ser comparadas diretamente.

> A transformação deve ser aplicada aos inputs e outputs conhecidos pela mesma regra. Transformar somente a entrada de teste quebraria a relação espacial aprendida nos exemplos.

---

## 3. Conjunto mínimo de transformações

A versão recomendada usa quatro transformações:

| Nome | Operação | Inversa |
| :--- | :--- | :--- |
| Identidade | Não altera o grid | Identidade |
| Flip horizontal | Inverte a ordem das colunas | Flip horizontal |
| Flip vertical | Inverte a ordem das linhas | Flip vertical |
| Transposição | Troca linhas por colunas | Transposição |

Essas operações são simples porque todas são involuções: aplicar a mesma operação duas vezes recupera o grid original.

Para um grid `G`:

```text
identidade(G)       = G
flip_horizontal(G)  = inverter cada linha de G
flip_vertical(G)    = inverter a ordem das linhas de G
transpose(G)        = trocar G[i][j] por G[j][i]
```

A transposição de um grid retangular troca sua altura e largura. Isso é esperado e não deve ser tratado como erro. O output transformado terá a forma correspondente ao input transformado e voltará ao formato original depois da transposição inversa.

### 3.1 Extensão opcional para oito transformações

Depois de validar a versão de quatro transformações, o conjunto pode ser estendido para as oito simetrias do grupo `D8`:

- identidade;
- rotações de 90, 180 e 270 graus;
- flip horizontal;
- flip vertical;
- reflexão na diagonal principal;
- reflexão na diagonal secundária.

A lógica do pipeline permanece igual. Apenas aumenta o número de chamadas ao modelo.

---

## 4. Geração das respostas

Para cada transformação `T`, o modelo realiza uma inferência independente usando **decodificação gulosa**.

Na decodificação gulosa, em cada passo é escolhido o token com maior probabilidade local:

$$
 token_t = \arg\max_{v} P(v \mid token_{<t}, T(C))
$$

A geração continua até o token de fim (`eos`) ou até atingir o limite máximo de tokens.

Com quatro transformações, espera-se uma predição por transformação:

```text
Tarefa original       -> y_identity
Flip horizontal       -> y_horizontal
Flip vertical         -> y_vertical
Transposição          -> y_transpose
```

As respostas das visões transformadas ainda estão em orientações diferentes. Antes de votar, é obrigatório aplicar a inversa correspondente:

```text
y_identity   -> identity(y_identity)
y_horizontal -> flip_horizontal(y_horizontal)
y_vertical   -> flip_vertical(y_vertical)
y_transpose  -> transpose(y_transpose)
```

Depois dessa etapa, todas as respostas devem descrever uma matriz na orientação original.

---

## 5. Votação global

Com as respostas normalizadas, realiza-se uma votação por igualdade exata do grid.

Exemplo:

```text
Resposta normalizada 1: [[1, 0], [0, 1]]
Resposta normalizada 2: [[1, 0], [0, 1]]
Resposta normalizada 3: [[0, 1], [1, 0]]
Resposta normalizada 4: [[1, 0], [0, 1]]
```

Nesse caso:

```text
[[1, 0], [0, 1]] -> 3 votos
[[0, 1], [1, 0]] -> 1 voto
```

A primeira matriz é a resposta final.

Para uma competição que permite duas tentativas (`pass@2`), as duas matrizes mais frequentes podem ser submetidas. É importante eliminar duplicatas antes de selecionar as duas respostas, pois o mesmo grid pode aparecer em várias perspectivas.

### 5.1 Empates

Com poucas transformações, empates são possíveis. Uma política simples e determinística é usar esta ordem de prioridade:

1. maior número de votos;
2. predição da identidade;
3. ordem fixa das transformações.

A preferência pela identidade não prova que essa visão é melhor. Ela apenas fornece um desempate estável e segue a ideia usada em estratégias de votação hierárquica: conservar a resposta da tarefa original quando não há evidência suficiente para preferir outra.

Outra opção é usar o score de confiança do modelo como desempate, caso a API disponibilize log-probabilidades. Isso é opcional e não é necessário para a primeira implementação.

---

## 6. Pseudocódigo

```python
TRANSFORMS = [
    ("identity", identity, identity),
    ("flip_horizontal", flip_horizontal, flip_horizontal),
    ("flip_vertical", flip_vertical, flip_vertical),
    ("transpose", transpose, transpose),
]


def predict_with_augmentation(task, model):
    normalized_predictions = []

    for name, transform, inverse_transform in TRANSFORMS:
        augmented_task = transform_task(task, transform)
        augmented_output = greedy_decode(model, augmented_task)
        output = inverse_transform(augmented_output)

        normalized_predictions.append({
            "transform": name,
            "output": output,
        })

    ranked_outputs = rank_by_exact_frequency(normalized_predictions)
    return ranked_outputs[:2]
```

A função `transform_task` deve transformar todos os grids da tarefa, e não somente a entrada de teste. A função `greedy_decode` deve apenas gerar a saída para a entrada de teste da tarefa transformada.

Uma versão mais explícita do ranking seria:

```python
def rank_by_exact_frequency(predictions):
    groups = {}

    for item in predictions:
        key = serialize_grid(item["output"])
        if key not in groups:
            groups[key] = {
                "output": item["output"],
                "votes": 0,
                "has_identity": False,
            }

        groups[key]["votes"] += 1
        groups[key]["has_identity"] |= item["transform"] == "identity"

    return sorted(
        groups.values(),
        key=lambda item: (
            -item["votes"],
            -int(item["has_identity"]),
        ),
    )
```

Na prática, a serialização usada como chave deve ser inequívoca. JSON compacto é uma opção adequada para grids compostos apenas por inteiros:

```python
key = json.dumps(output, separators=(",", ":"))
```

---

## 7. Exemplo completo

Considere quatro predições produzidas pelo modelo:

| Visão | Saída produzida na visão transformada | Saída após aplicar a inversa |
| :--- | :--- | :--- |
| Identidade | `A` | `A` |
| Flip horizontal | `flip(A)` | `A` |
| Flip vertical | `B` transformado | `B` |
| Transposição | `transpose(A)` | `A` |

Depois da normalização:

```text
A -> 3 votos
B -> 1 voto
```

O sistema escolhe `A` como primeira resposta. Se houvesse uma segunda resposta permitida, `B` seria o segundo palpite.

O ponto importante é que as respostas `flip(A)` e `transpose(A)` não são comparadas diretamente com `A`. Primeiro é preciso desfazer a transformação que foi aplicada à tarefa.

---

## 8. Relação com os artigos

### 8.1 Relação com o artigo do MIT

A estratégia se aproxima da etapa de **inferência aumentada** do artigo do MIT:

- transforma a tarefa;
- gera uma resposta em cada visão;
- aplica a transformação inversa;
- agrega as respostas por votação.

A diferença é que esta versão simplificada não implementa:

- TTT;
- treinamento de um LoRA por tarefa;
- dataset `D_TTT` com tarefas *leave-one-out*;
- múltiplas permutações da ordem dos exemplos;
- votação intra-transformação;
- candidatos sintéticos por maioria de linhas ou colunas.

### 8.2 Relação com o artigo *The LLM ARChitect*

A ideia de explorar perspectivas diferentes também aparece no *The LLM ARChitect*, mas o mecanismo é diferente:

| Aspecto | Estratégia simples | LLM ARChitect |
| :--- | :--- | :--- |
| Geração | Uma saída gulosa por transformação | DFS com vários candidatos por visão |
| Número de respostas | Aproximadamente uma por transformação | Possivelmente várias por transformação |
| Seleção | Frequência global | Soma de log-probabilidades (`AugScore`) |
| TTT | Não | Não no mesmo formato por tarefa |
| Complexidade | Baixa | Maior |

A estratégia simples troca parte do poder do `AugScore` e da DFS por uma implementação menor e mais transparente.

---

## 9. O que esta estratégia não garante

A votação não garante que a resposta mais frequente seja correta. Ela funciona melhor quando:

- diferentes perspectivas levam o modelo à mesma solução;
- erros variam entre as perspectivas;
- a transformação é aplicada corretamente;
- a saída é parseada sem alterar o grid;
- respostas equivalentes são serializadas exatamente da mesma forma.

Se o mesmo erro aparecer em todas as visões, a votação apenas reforçará esse erro. Da mesma forma, se cada visão produzir uma resposta diferente, não haverá consenso forte.

Por isso, essa técnica deve ser vista como uma forma de **autoconsistência geométrica**, não como uma verificação formal da solução.

---

## 10. Ordem recomendada de implementação

### Etapa 1 — Linha de base

Implementar apenas:

```text
tarefa original -> modelo -> uma resposta
```

Guardar a resposta e medir a acurácia.

### Etapa 2 — Transformações e inversas

Implementar e testar separadamente:

- identidade;
- flip horizontal;
- flip vertical;
- transposição.

Para cada transformação, verificar que:

$$
T^{-1}(T(G)) = G
$$

para grids quadrados e retangulares.

### Etapa 3 — Inferência aumentada

Rodar o modelo em cada uma das quatro versões e trazer as respostas para a orientação original.

### Etapa 4 — Votação

Agrupar as respostas normalizadas por igualdade exata e selecionar a mais frequente.

### Etapa 5 — `pass@2`

Selecionar as duas respostas mais frequentes, removendo duplicatas.

### Etapa 6 — Extensões opcionais

Somente depois de validar a versão básica, considerar:

- rotações de 90, 180 e 270 graus;
- duas ordens diferentes dos exemplos;
- votação hierárquica;
- probabilidades ou log-probabilidades como desempate;
- permutações de cores.

---

## 11. Custo computacional

Com quatro transformações, o número de chamadas de geração é aproximadamente quatro vezes maior que na linha de base:

$$
C_{aug} \approx 4 \times C_{base}
$$

Isso torna a abordagem simples de testar, mas ainda pode ser caro em modelos grandes. As chamadas podem ser processadas em lote quando as sequências forem compatíveis e houver memória suficiente.

O custo adicional da votação é pequeno: depois das respostas geradas, basta aplicar as inversas e contar grids iguais.

---

## 12. Resumo

A implementação mínima recomendada é:

```text
1. Criar 4 versões da tarefa: identidade, flip horizontal,
   flip vertical e transposição.
2. Gerar uma resposta gulosa para cada versão.
3. Aplicar a transformação inversa a cada resposta.
4. Agrupar respostas iguais.
5. Selecionar a mais frequente ou as duas mais frequentes.
```

Essa abordagem é consideravelmente mais simples que a combinação de TTT, DFS e `AugScore`, mas preserva a intuição principal dos artigos: uma regra correta tende a continuar fazendo sentido quando a tarefa é observada sob diferentes perspectivas.
