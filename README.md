# pas-mri-extractor

Локальный пайплайн для извлечения структурированных признаков PAS из текстовых
описаний МРТ. Проект валидирует JSON через Pydantic, рассчитывает клинические
риски и поддерживает Streamlit-интерфейс для просмотра результата.

`extracted_features.invasion` хранит основной извлеченный тип PAS.
`suspicion` хранит отдельный safety-блок для неопределенных формулировок
вроде "нельзя исключить percreta" и не подменяет `invasion.type`.

## Установка

```bash
git clone https://github.com/ESBehtev/pas-mri-extractor.git
cd pas-mri-extractor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Для llama.cpp backend на сервере должен быть установлен `llama-cpp-python`.

## Актуальные модели

Модели описаны в `configs/models.yaml`.

Production default:

```text
qwen3_27b_q4_k_m_gguf
```

Дополнительная GGUF-модель:

```text
qwen3_6_35b_a3b_gguf
```

Fallback без GGUF:

```text
qwen_2_5_7b
```

Локальный dry-run только валидирует конфиг и показывает resolved path.
Реальная загрузка GGUF локально не выполняется.

```bash
PYTHONPATH=src python run_single.py --model qwen3_27b_q4_k_m_gguf --dry-run-model-config
PYTHONPATH=src python run_single.py --model qwen3_6_35b_a3b_gguf --dry-run-model-config
```

## Скачивание GGUF на сервере

27B Q4_K_M:

```bash
mkdir -p ~/pas-mri-extractor/models/Qwen3.6-27B-GGUF
cd ~/pas-mri-extractor/models/Qwen3.6-27B-GGUF
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir .
test -s ~/pas-mri-extractor/models/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf
```

35B-A3B UD-Q4_K_M:

```bash
mkdir -p ~/pas-mri-extractor/models/Qwen3.6-35B-A3B-GGUF
cd ~/pas-mri-extractor/models/Qwen3.6-35B-A3B-GGUF
hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir .
test -s ~/pas-mri-extractor/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

## Запуск на сервере

Streamlit:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py --server.port 8501
```

Smoke test GGUF:

```bash
PYTHONPATH=src python -m pas_mri_extractor.smoke_test_llama_cpp --model qwen3_27b_q4_k_m_gguf
PYTHONPATH=src python -m pas_mri_extractor.smoke_test_llama_cpp --model qwen3_6_35b_a3b_gguf
```

Batch eval:

```bash
PYTHONPATH=src python -m pas_mri_extractor.batch_eval \
  --config configs/eval_qwen.yaml \
  --model qwen3_27b_q4_k_m_gguf

PYTHONPATH=src python -m pas_mri_extractor.batch_eval \
  --config configs/eval_qwen.yaml \
  --model qwen3_6_35b_a3b_gguf
```

Single report:

```bash
PYTHONPATH=src python run_single.py \
  --model qwen3_27b_q4_k_m_gguf \
  --text-file examples/sample_mri.txt
```

## Batch eval YAML

Eval сохраняет raw output, parsed JSON, diff и summary в `outputs/eval/<run_name>/`.
Case может использовать `text_file` или inline `text`, но не оба поля.

```yaml
run_name: qwen_server_eval
output_dir: outputs/eval
model: qwen3_27b_q4_k_m_gguf

cases:
  - id: inline_case_001
    text: |
      МРТ малого таза. Беременность 34 недели.
      Плацента по передней стенке, признаков врастания не выявлено.
    expected:
      extracted_features:
        invasion:
          type: none
          confidence: absent
```

## JSON mode

Для llama_cpp моделей включен backend-level JSON mode:

```yaml
output:
  response_format:
    type: json_object
  enforce_json: true
```

Если установленная версия `llama-cpp-python` не поддерживает `response_format`,
backend пишет warning и повторяет генерацию без этого аргумента. Дальше остаются
обычные safety net: prompt instruction, robust JSON extraction, retry и Pydantic
validation.

## Suspicion

`suspicion` отделяет клиническое подозрение от основного вывода:

```json
{
  "suspicion": {
    "highest_suspected_extent": "none",
    "percreta_suspicion": "absent",
    "bladder_serosa_suspicion": "absent",
    "rationale": []
  }
}
```

Старые payload без `suspicion` валидируются с default-блоком. Подозрение на
percreta учитывается как safety red flag в scoring, но не меняет
`extracted_features.invasion.type`.

## OOM на RTX 3090

При CUDA OOM уменьшайте параметры в `configs/models.yaml`:

- `runtime.n_ctx`: `8192` -> `4096`;
- `generation.max_tokens` и `generation.max_new_tokens`: ниже `2048`;
- `runtime.n_gpu_layers`: `-1` -> `60`, затем ниже;
- `runtime.n_batch`: `512` -> `256`.

Проверка GPU:

```bash
watch -n 1 nvidia-smi
```

Во время smoke test или batch eval должен расти VRAM usage процесса Python.

## Локальные проверки

```bash
PYTHONPATH=src python -m compileall run_single.py src app
PYTHONPATH=src python run_single.py --model qwen3_27b_q4_k_m_gguf --dry-run-model-config
PYTHONPATH=src python run_single.py --model qwen3_6_35b_a3b_gguf --dry-run-model-config
```
