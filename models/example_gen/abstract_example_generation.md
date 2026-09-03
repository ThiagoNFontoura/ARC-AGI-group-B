# Geração de Exemplos Sintéticos para ARC-AGI

## 1. Visão geral

O método de geração de exemplos foi projetado para aumentar uma tarefa ARC com dados supervisionados sintéticos antes de uma etapa posterior de inferência. Ele não atualiza os parâmetros do Transformer, não realiza fine-tuning e não executa treinamento supervisionado. Em vez disso, usa um modelo mais forte como uma "trapaça" controlada: apresenta a ele os exemplos rotulados disponíveis, solicita que infira a regra da tarefa e usa essa resposta para criar novos pares de entrada e saída que sigam a transformação identificada. O modelo usado posteriormente para resolver a task não é usado para gerar os exemplos, pois, se ele já conseguisse inferir corretamente a regra para gerar dados válidos, poderia responder diretamente.

A unidade de processamento é uma task ARC individual. Quando o programa recebe uma pasta, cada arquivo JSON válido é processado separadamente e em ordem alfabética. Para cada task, o sistema constrói um prompt, faz uma inferência no modelo e salva o resultado como um novo arquivo com o sufixo `-plus.json`.

## 2. Construção do contexto

Para cada task, o prompt inclui todos os pares de treino disponíveis, contendo suas entradas e saídas conhecidas. As entradas de teste também são incluídas, mas suas saídas são removidas para evitar que o modelo receba a resposta correta durante a geração. Usar todos os pares evita que exemplos relevantes sejam ignorados por uma limitação arbitrária aos três primeiros.

Se $C$ representa o contexto de treino e $X_{test}$ representa as entradas de teste sem seus rótulos, o modelo recebe:

$$
(C, X_{test}) \longrightarrow (r, G, \hat{Y}_{test})
$$

onde:

- $r$ é uma explicação curta da regra inferida;
- $G$ é o conjunto de exemplos de treino gerados;
- $\hat{Y}_{test}$ são as previsões para as entradas de teste.

O prompt exige que as matrizes geradas contenham inteiros e sejam retangulares. Também instrui o modelo a preservar altura e largura quando essas dimensões forem constantes nos exemplos originais, evitando que a geração altere a estrutura da tarefa sem evidência.

## 3. Geração em uma única chamada

Para cada task, o modelo realiza uma única inferência gulosa por meio da API `generate_content`. A resposta deve ser um objeto JSON estrito com quatro campos:

```json
{
  "logic_explanation": "explicação breve da regra",
  "generated_train": [
    {"input": [[...]], "output": [[...]]}
  ],
  "predicted_test_outputs": [[[...]]],
  "validation": {
    "original_train": [
      {"index": 0, "passed": true, "reason": "..."}
    ],
    "generated_train": [
      {"index": 0, "passed": true, "reason": "..."}
    ]
  }
}
```

O parâmetro `generated_examples` define quantos pares adicionais devem aparecer em `generated_train`. Ele controla a quantidade de exemplos dentro da resposta, e não o número de requisições. Com `generated_examples = 10`, o sistema ainda faz apenas uma chamada por task, mas solicita dez novos pares nessa chamada.

Assim, para $N$ tasks e sem retries, o número esperado de requisições é:

$$
R = N
$$

O número de exemplos solicitados é aproximadamente:

$$
E = N \times generated\_examples
$$

Aumentar `generated_examples` tende a aumentar os tokens de saída e o tempo de uma chamada, mas não multiplica diretamente as requisições à API.

## 4. Validação da resposta

Depois da resposta, o programa extrai o objeto JSON e valida localmente sua estrutura. A explicação da regra precisa existir e não pode estar vazia. O campo `generated_train` precisa ser uma lista com exatamente a quantidade solicitada de exemplos. Cada exemplo deve conter grids retangulares de inteiros e deve respeitar as propriedades marcadas como constantes nos exemplos de treino.

Essa validação não faz uma segunda chamada ao modelo e não prova formalmente que os exemplos gerados estão corretos. Ela verifica o formato e as invariantes estruturais observadas nos dados. A correção semântica dos pares ainda depende da regra inferida pelo próprio modelo.

Essa barreira automática complementa o processo anterior, no qual a correção era conferida principalmente por inspeção manual: um renderer separado produzia imagens dos inputs e outputs de algumas tasks ou conjuntos de tasks, e a consistência era analisada visualmente. Como não há um modelo gratuito mais forte disponível no AI Studio para validar independentemente a compreensão do `gemini-3.7-flash`, essa inspeção manual continua sendo útil, mas agora é complementada por uma verificação automática.

Se a resposta for inválida, a task falha localmente. Com `transient_retry_attempts = 0`, nenhuma nova requisição é feita para tentar corrigir a resposta. Erros da API, como indisponibilidade do modelo ou quota excedida, também encerram apenas aquela task e não geram retries automáticos.

## 5. Montagem da task aumentada

Quando a resposta é válida, o arquivo `-plus.json` é montado com:

1. o nome da task original;
2. a explicação da regra inferida;
3. todos os exemplos de treino originais;
4. os exemplos de treino sintéticos gerados pelo modelo;
5. as entradas de teste sem suas saídas verdadeiras;
6. o nome do arquivo de origem e o horário de geração.

Formalmente, se $C$ são todos os exemplos originais e $G$ é o conjunto gerado, o treino da task aumentada é:

$$
C^+ = C \cup G
$$

A task aumentada serve como uma nova representação supervisionada do mesmo problema, mas não deve ser confundida com uma garantia de que a regra foi descoberta corretamente. O objetivo experimental é medir se exemplos produzidos pelo modelo mais forte, especialmente o `gemini-3.7-flash`, melhoram a acurácia do solver mais fraco; essa melhoria ainda precisa ser avaliada em testes mais profundos. Exemplos sintéticos incorretos podem reforçar uma interpretação errada e prejudicar a inferência posterior.

O campo `predicted_test_outputs` é uma previsão auxiliar para as entradas de teste. Ele não representa exemplos supervisionados e não é usado na montagem atual do dataset aumentado: o arquivo `-plus.json` preserva somente as entradas de teste sem rótulos. A geração de exemplos e a resolução da task são etapas separadas.

## 6. Custo computacional e uso da API

O custo principal é a inferência do modelo para cada task. No caso configurado para o `example-gen`, cada task faz uma chamada ao modelo `gemini-3.7-flash` com nível alto de pensamento. O custo em tokens inclui tanto o prompt, que contém os exemplos da task, quanto a resposta, que contém a explicação, os pares sintéticos e os registros de validação solicitados.

Com $N$ tasks processadas e retries desativados:

$$
\text{chamadas ao modelo} = N
$$

Uma falha de API pode consumir uma requisição mesmo sem produzir um arquivo, pois a chamada chega ao provedor antes de retornar o erro. Por isso, a quantidade de arquivos `-plus.json` não é uma medida confiável do número de requisições consumidas.

A configuração atual prioriza economia de chamadas: `transient_retry_attempts` é zero e não há timeout local aplicado ao pensamento do modelo. Os limites de requisições configurados no JSON são referências operacionais; o controle efetivo da quota pertence ao provedor da API.

## 7. Análise de invariantes

As propriedades são analisadas separadamente para inputs e outputs. Uma propriedade só é marcada como constante quando possui exatamente o mesmo valor em todos os grids válidos daquele lado. Não se exige que uma propriedade tenha o mesmo valor entre input e output, pois uma regra pode transformar, por exemplo, um input $3 \times 3$ em um output $1 \times 1$.

As propriedades observadas incluem:

- altura e largura do grid;
- conjunto de cores presentes;
- contagem de cada cor;
- cor de fundo estimada;
- quantidade de células que não pertencem ao fundo;
- número de componentes conectados em quatro direções;
- simetria horizontal e vertical exatas.

As propriedades constantes são enviadas no prompt como restrições para os exemplos novos. As propriedades não constantes não são congeladas: o modelo deve analisar como elas variam entre os exemplos e usar essa variação para inferir a regra. Essa separação delimita a geração sem impor que toda task mantenha o mesmo tamanho, número de objetos ou distribuição de cores quando os dados demonstram o contrário.

## 8. Perguntas ainda abertas

As decisões anteriores respondem às perguntas sobre objetivo, formalização, uso dos exemplos, validação, invariantes, delimitação da geração e quantidade solicitada. Permanecem estas questões:

**11. Registro de falhas**

Decisão: sim. O example-gen salva `relatorio[ID].json` junto aos outputs. Para uma task individual, `ID` é o nome da task; para uma pasta, `ID` é o nome da pasta. O relatório registra modelo, nível de pensamento, quantidade solicitada, retries, status por task, arquivos produzidos e erros.

**12. Reprodutibilidade**

Decisão provisória: aceitar variação entre execuções. A diversidade de exemplos é útil, e fixar uma seed antes de definir um protocolo de comparação poderia reduzir essa diversidade. A comparação de acurácia deve registrar modelo, configuração, task set e timestamp; determinismo pode ser adicionado depois para ablações controladas.

**13. Critério de sucesso**

Decisão: o modelo deve inferir uma regra de input para output, testá-la nos pares originais e gerados e retornar flags por exemplo. O código também verifica formato e invariantes. Exemplos com `valid: false` são ignorados pelo solver. O sucesso experimental completo exige essa validação e, posteriormente, medir o efeito na acurácia do solver.

**14. Limites fora do escopo**

Decisão: não. Uma propriedade não constante pode ser precisamente a chave da regra. O sistema só congela propriedades constantes; variações de cores, objetos, conectividade, dimensões e simetria permanecem disponíveis para a inferência.

**15. Nome do método**

Decisão: `exemplar synthesis` é o nome curto preferido; em português, “síntese de exemplos”.

**16. Validação semântica futura**

Decisão: o modelo gerador deve montar a regra, aplicá-la a todos os pares e retornar os resultados de validação. A inspeção manual por imagens, usada antes, permanece como auditoria. Se mais de 20% dos exemplos extras falharem, todos os extras recebem `valid: false`; caso contrário, somente os falhos são descartados pelo solver. A margem de 20% é um compromisso inicial razoável: abaixo disso, preserva exemplos bons; acima disso, sugere que a regra inteira está pouco confiável.

**17. Separação dos resultados**

Decisão: salvar ambos. O `-plus.json` contém a explicação, as flags por exemplo e a análise de invariantes com cada propriedade listada como constante quando aplicável. O solver recebe apenas os exemplos sem `valid: false`; o relatório conserva o histórico da geração e da validação.

## 9. Característica principal

A geração de exemplos desloca o custo para uma única inferência por task e usa a capacidade do modelo de explicar uma regra para produzir dados adicionais que podem ser consumidos por um solver posterior. Sua vantagem é transformar uma pequena quantidade de demonstrações em um contexto de treino mais rico sem alterar os pesos do modelo.

Seu principal risco é a propagação de erros: se a regra inferida estiver errada, os exemplos gerados podem ser consistentes entre si e ainda assim representar a transformação incorreta. Portanto, o método aumenta a quantidade de evidência disponível, mas não substitui uma verificação formal da regra nem garante que os exemplos sintéticos sejam semanticamente corretos.
