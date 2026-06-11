# pas-mri-extractor

Локальный пайплайн для извлечения структурированных признаков PAS из текстовых
описаний МРТ. Проект валидирует JSON через Pydantic, рассчитывает клинические
риски и поддерживает Streamlit-интерфейс для просмотра результата.

`extracted_features.invasion` хранит основной извлеченный тип PAS.
`suspicion` хранит отдельный safety-блок для неопределенных формулировок
вроде "нельзя исключить percreta" и не подменяет `invasion.type`.

## Current Stage Architecture

Текущий публичный Streamlit-запуск сохраняется:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py --server.port 8501
```

Параллельно добавлен минимальный синхронный stage-based слой без LangGraph,
FastAPI и очередей:

- `ExtractorStage`: тонкая обёртка над текущим extraction pipeline. Она отвечает
  за структурированное извлечение PAS JSON из текста МРТ и не меняет prompt,
  schema или поведение существующего extractor.
- `RiskPredictionStage`: стадия расчёта score, `predicted_risks`,
  `recommendation` и `computed_rationale` на основе уже извлечённого JSON.
  Сейчас она использует существующую rule-based функцию
  `scoring.normalize_mri_result()`, поэтому численные значения и readiness logic
  остаются прежними.

Программная точка входа:

```python
from pas_mri_extractor.orchestrator import run_case_pipeline

stage_results = run_case_pipeline(text, model_id="qwen3_6_35b_a3b_gguf")
```

`PipelineContext` уже хранит `source_text`, optional `conclusion_text`,
`extracted_result`, `predicted_risks`, `evidence` и `metadata`, чтобы следующими
шагами можно было добавить более сложный prognosis слой без изменения
extraction schema.

Planned stages/services:

- `ValidationStage`;
- ML/calibrated prognosis model;
- `ClinicalSummaryStage`;
- `CaseChatService` поверх финального case context;
- MRI/DICOM segmentation/classification pipeline.

## Prompt Registry

Активный extractor prompt по-прежнему хранится в `configs/prompt.yaml`.
Это сохраняет совместимость существующего `build_prompt()` и не меняет
поведение extraction.

Для stage-based архитектуры добавлен prompt registry:

- `extractor` резолвится в текущий активный `configs/prompt.yaml`;
- `configs/prompts/extractor.yaml` является documented alias и не используется
  runtime extraction;
- `configs/prompts/risk_prediction.yaml` является experimental LLM risk
  prediction prompt для терминальных экспериментов и не подключён к основному
  `run_case_pipeline()`;
- `configs/prompts/risk_prediction.example.yaml` оставлен как planned/example
  prompt и не используется текущим rule-based `RiskPredictionStage`;
- `configs/prompts/clinical_summary.example.yaml` является planned clinical
  summary prompt;
- `configs/prompts/case_chat.example.yaml` является planned case chat prompt.

Рекомендуемая структура `configs/`:

- `configs/prompts/`: stage prompt registry entries и future prompt configs;
- `configs/prompt.yaml`: active extractor prompt, сохранён для backward
  compatibility;
- `configs/models.yaml`: model registry;
- `configs/rules.yaml`: rule baseline config;
- `configs/risk_score.yaml`: deterministic score/risk config;
- `configs/benchmark_pas20.yaml`: PAS20 benchmark config;
- `configs/eval_*.yaml`: YAML batch eval configs.

В будущем разные стадии смогут использовать разные prompt configs и разные
модели без изменения clinical schema.

Terminal-only LLM risk prediction experiment:

```python
from pas_mri_extractor.orchestrator import run_risk_prediction_experiment
from pas_mri_extractor.pipeline import get_cached_model

loaded_model = get_cached_model("qwen3_6_35b_a3b_gguf")
stage_result = run_risk_prediction_experiment(
    text=mri_text,
    extracted_result=extracted_json,
    model_id="qwen3_6_35b_a3b_gguf",
    loaded_model=loaded_model,
)
```

Если `runner` или `loaded_model` передан явно, экспериментальная стадия не
загружает GGUF повторно. Без них fallback использует process-level
`get_cached_model()`.

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

Current default:

```text
qwen3_6_35b_a3b_gguf
```

Дополнительная GGUF-модель:

```text
qwen3_27b_q4_k_m_gguf
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

## Ретроспективный evaluation pipeline

Экспорт третьего листа Excel из корня проекта в CSV и JSONL:

```bash
PYTHONPATH=src python scripts/export_excel_dataset.py
```

Если Excel-файл нужно указать явно:

```bash
PYTHONPATH=src python scripts/export_excel_dataset.py \
  --xlsx "Копия Итог 2 этап_v2 (00000002).xlsx"
```

Результат сохраняется в:

```text
data/evaluation/dataset_sheet3.csv
data/evaluation/dataset_sheet3.jsonl
```

Построение ретроспективных gold-полей в новых файлах без изменения исходного
`dataset_sheet3.csv`:

```bash
PYTHONPATH=src python scripts/build_gold_dataset.py
```

Результат сохраняется в:

```text
data/evaluation/dataset_sheet3_gold.csv
data/evaluation/dataset_sheet3_gold.jsonl
```

Проверка пайплайна без LLM на rule baseline:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py --use-rules --limit 10
```

Оценка через LLM на сервере:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --run-llm \
  --model qwen3_6_35b_a3b_gguf \
  --limit 10
```

Результаты:

```text
outputs/evaluation/predictions.jsonl
outputs/evaluation/metrics.json
```

Если названия колонок МРТ отличаются от ожидаемых, передайте их явно:

```bash
PYTHONPATH=src python scripts/evaluate_predictions.py \
  --use-rules \
  --text-columns "МРТ_Описание" "МРТ_Заключение" \
  --limit 10
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
