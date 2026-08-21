# Como rodar o ARC solver

Execute todos os comandos a partir da raiz do repositorio

## Antes de rodar

As tasks devem estar em:

```text
data\small_training\
```

`small_training` e apenas um exemplo. Troque esse nome nos comandos abaixo pelo nome da pasta que voce quer processar, por exemplo `test-10-tasks` ou `tasks-to-solve`.

A chave da API deve estar configurada no arquivo `.env`:

```env
GEMMA_API_KEY=sua_chave_aqui
```

## 1. Rodar o solver JSON

Envia os dados JSON das tasks para o modelo e salva o relatorio JSON:

```powershell
python -m models.image_baseline_model.main small_training
```

Troque `small_training` pelo nome da pasta desejada.

## 2. Renderizar as tasks em PNG

Gera somente as imagens PNG das tasks, sem chamar o modelo:

```powershell
python -m models.image_baseline_model.main small_training --render-only
```

As imagens sao salvas em uma pasta de output com o nome da pasta de origem e um indice.

## 3. Rodar o solver PNG

Renderiza as tasks e depois envia as imagens para o solver. Esse comando tambem executa o solver JSON antes do solver PNG, conforme o fluxo atual:

```powershell
python -m models.image_baseline_model.main small_training --render-images
```

Troque `small_training` pelo nome da pasta desejada.

## Saidas

Os resultados ficam em:

```text
output\small_training_001\
```

Exemplos de arquivos gerados:

```text
output\small_training_001\small_training_001-json.json
output\small_training_001\small_training_001-image.json
output\small_training_001\images\
```

O numero do indice aumenta quando ja existem relatorios com indice anterior para a mesma pasta de tasks.
