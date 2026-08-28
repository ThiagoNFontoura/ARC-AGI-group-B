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

O `task-gen` usa `gemini-3.7-flash` pela configuracao em `models/task_gen/task_gen_config.json`.

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
python -m models.task_gen.main data/minhas-tasks
python -m models.task_gen.main data/minhas-tasks/3aa6fb7a.json
```

Cada resultado e salvo em `output/task-gen/` com o sufixo `-plus.json`, por exemplo:

```text
output/task-gen/3aa6fb7a-plus.json
```

O task-gen respeita no maximo 5 requisicoes por minuto e 20 por dia (UTC). Cada tentativa, incluindo retries apos erros temporarios, conta para esses limites. O uso fica registrado em `output/task-gen/.request_usage.json` para continuar sendo controlado entre execucoes.

O numero de exemplos gerados pode ser alterado pela configuracao em `models/task_gen/task_gen_config.json` ou sobrescrito na execucao:

```powershell
python -m models.task_gen.main data/minhas-tasks --generated-examples 5
```

Tasks que ja terminam em `-plus.json` sao ignoradas quando uma pasta e processada.
