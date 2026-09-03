# ARC-AGI G1

## Preparar o ambiente

No PowerShell, na raiz do projeto:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Crie um arquivo `.env` com a chave da API:

```env
GEMMA_API_KEY=sua-chave
```

Use as credenciais no formato fornecido pelo seu ambiente, sem alterar o valor nem adicionar `Bearer`.

Nunca coloque a chave diretamente no codigo ou no Git.

Opcionalmente, defina o modelo usado pelo solver:

```env
GEMMA_MODEL=gemini-3.5-flash-lite
GEMMA_VALIDATOR_MODEL=gemini-3.5-flash-lite
```

O `example-gen` usa `gemini-3.7-flash` pela configuracao em `models/example_gen/example_gen_config.json` e solicita o nivel alto de pensamento.

## Estrutura das tasks

Coloque os arquivos JSON ARC em uma pasta dentro de `data/`, por exemplo:

```text
data/minhas-tasks/*.json
```

Cada arquivo deve conter as listas `train` e `test`.

## Solver textual

Executa o modelo usando os dados JSON, sem imagens:

```powershell
python -m models.baseline_model.main minhas-tasks
```

O relatório e salvo em `output/<pasta>_baseline_output.json`.

## Solver com imagens

Executa o solver com as imagens renderizadas:

```powershell
python -m models.image_baseline_model.main minhas-tasks --render-images
```

O estagio JSON do solver nao possui limite de tempo. Para processar apenas uma task:

```powershell
python -m models.image_baseline_model.main training --task-file 007bbfb7.json --no-strong-validate
```

Para somente renderizar as imagens, sem chamar o modelo:

```powershell
python -m models.image_baseline_model.main minhas-tasks --render-only
```

Os resultados ficam em `output/<pasta>_<numero>/`, e as imagens em sua subpasta `images/`.

Opcoes uteis:

```powershell
python -m models.image_baseline_model.main minhas-tasks --render-workers 4 --no-strong-validate
```

## Gerar tasks aumentadas

Gera exemplos de treino adicionais para uma task ou para todas as tasks de uma pasta:

```powershell
python -m models.example_gen.main data/minhas-tasks
python -m models.example_gen.main data/minhas-tasks/3aa6fb7a.json
```

Cada resultado e salvo em `output/example-gen/` com o sufixo `-plus.json`, por exemplo:

```text
output/example-gen/3aa6fb7a-plus.json
```

O example-gen foi configurado para no maximo 5 requisicoes por minuto e 10 por dia (UTC), sem retries automaticos. O programa nao interrompe uma chamada por tempo de pensamento: aguarda a resposta do modelo. Os campos de limite sao informativos; o controle de cota deve ser feito no provedor.

O numero de exemplos gerados pode ser alterado pela configuracao em `models/example_gen/example_gen_config.json` ou sobrescrito na execucao:

```powershell
python -m models.example_gen.main data/minhas-tasks --generated-examples 5
```

O example-gen tambem salva `relatorio[ID].json` em `output/example-gen/`. O relatorio registra o status de cada task e os resultados da validacao por exemplo. Cada par de treino recebe a flag `valid`; os solvers ignoram exemplos com `valid: false`. Se mais de 20% dos exemplos novos falharem na validacao, todos os exemplos novos daquela task sao marcados como invalidos.

Tasks que ja terminam em `-plus.json` sao ignoradas quando uma pasta e processada.
