# pas-mri-extractor

LLM-пайплайн для извлечения признаков PAS (placenta accreta spectrum) из текстовых описаний МРТ.

Проект:
- извлекает структурированные признаки из МРТ,
- валидирует результат через Pydantic,
- рассчитывает клинический риск,
- формирует итоговую рекомендацию.

---

# Возможности

- LLM-based извлечение признаков
- Rule-based fallback через regex
- Валидация JSON через Pydantic
- Расчёт клинических рисков
- Конфиги моделей и prompt вынесены в YAML
- Примеры запуска в Jupyter Notebook

---

# Структура проекта

```text
pas-mri-extractor/
├── configs/
│   ├── models.yaml
│   ├── prompt.yaml
│   ├── risk_score.yaml
│   └── rules.yaml
│
├── examples/
│   └── sample_mri.txt
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   └── 01_example_runs.ipynb
│
├── src/
│   └── pas_mri_extractor/
│       ├── config.py
│       ├── extractor.py
│       ├── json_utils.py
│       ├── models.py
│       ├── prompts.py
│       ├── rules.py
│       ├── scoring.py
│       └── schemas.py
│
├── run_single.py
├── requirements.txt
└── pyproject.toml
```

---

# Установка

## Клонирование репозитория

```bash
git clone https://github.com/ESBehtev/pas-mri-extractor.git
cd pas-mri-extractor
```

---

## Создание виртуального окружения

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## Установка зависимостей

```bash
pip install -r requirements.txt
pip install -e .
```

---

# HuggingFace токен

В корне проекта создайте файл:

```text
.env
```

Содержимое:

```text
HF_TOKEN=your_huggingface_token
```

Токену достаточно read-доступа к моделям HuggingFace.

---

# Запуск

## Запуск Streamlit-интерфейса

```bash
streamlit run app/streamlit_app.py
```

---

## Запуск через LLM

```bash
python run_single.py --text-file examples/sample_mri.txt
```

Модель можно выбрать явно:

```bash
python run_single.py \
  --model qwen_3_6_35b_gguf \
  --text-file examples/sample_mri.txt
```

Или через переменную окружения:

```bash
PAS_MODEL=qwen_3_6_35b_gguf python run_single.py --text-file examples/sample_mri.txt
```

Проверить конфиг и наличие локальной модели без загрузки весов в GPU:

```bash
PYTHONPATH=src python run_single.py \
  --model qwen_3_6_35b_gguf \
  --dry-run-model-config
```

Для отладки JSON-ответа модели:

```bash
PYTHONPATH=src python run_single.py \
  --model qwen_3_6_35b_gguf \
  --text-file examples/sample_mri.txt \
  --print-raw-output
```

То же самое через env:

```bash
PAS_PRINT_RAW_OUTPUT=1 PYTHONPATH=src python run_single.py \
  --model qwen_3_6_35b_gguf \
  --text-file examples/sample_mri.txt
```

---

## Запуск только через regex-правила

```bash
python run_single.py --use-rules --text-file examples/sample_mri.txt
```

---

## Batch eval по YAML

Eval-контур читает YAML-конфиг, запускает модель или fixture-output,
валидирует результат через Pydantic-схему и сохраняет артефакты:

```text
outputs/eval/<run_name>/
├── raw/
├── parsed/
├── diff/
├── summary.csv
└── summary.json
```

Локальная dummy-проверка без загрузки модели:

```bash
PYTHONPATH=src python -m pas_mri_extractor.batch_eval \
  --config configs/eval_dummy.yaml \
  --fail-on-diff
```

Пример запуска LLM-eval на сервере:

```bash
PYTHONPATH=src python -m pas_mri_extractor.batch_eval \
  --config configs/eval_qwen.yaml \
  --model qwen_3_6_35b_gguf
```

Минимальная структура eval YAML:

```yaml
run_name: qwen_server_eval_001
output_dir: outputs/eval
model: qwen_3_6_35b_gguf

cases:
  - id: case_001
    text_file: examples/sample_mri.txt
    expected:
      extracted_features:
        invasion:
          type: none
          confidence: absent
```

Для проверки JSON-валидации без модели можно передать `raw_output` в case.
Если `raw_output` задан, модель не загружается для этого case.

---

## Передача текста напрямую

```bash
python run_single.py --text "Описание МРТ..."
```

---

# Работа через Jupyter Notebook

Откройте:

```text
notebooks/01_example_runs.ipynb
```

Рекомендуемый workflow:

```python
from pas_mri_extractor.models import load_llm

loaded_model = load_llm()
```

После этого модель не будет загружаться заново при каждом вызове.

Пример:

```python
from pas_mri_extractor.extractor import extract_mri_features

result = extract_mri_features(
    mri_text,
    loaded_model=loaded_model,
)
```

---

# Поддерживаемые модели

Модели задаются в:

```text
configs/models.yaml
```

Текущая модель по умолчанию:

```text
Qwen3.6-35B-A3B GGUF Q4_K_M
```

Основная модель:

```text
qwen_3_6_35b_gguf -> Qwen3.6-35B-A3B GGUF Q4_K_M
```

Локальный путь ожидается здесь:

```text
models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-Q4_K_M.gguf
```

Установить backend на сервере:

```bash
pip install llama-cpp-python
```

Скачать GGUF-модель на сервере:

```bash
hf download lmstudio-community/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --local-dir models/qwen3.6-35b-a3b-gguf
```

Запуск через GGUF backend:

```bash
PYTHONPATH=src python run_single.py \
  --model qwen_3_6_35b_gguf \
  --text-file examples/sample_mri.txt
```

Streamlit:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Fallback-модель Qwen2.5-7B сохранена:

```bash
python run_single.py \
  --model qwen_2_5_7b \
  --text-file examples/sample_mri.txt
```

---

# Что возвращает пайплайн

Результат содержит:

- версию JSON-схемы,
- информацию о случае,
- извлечённые MRI-признаки,
- evidence с положительными, неопределёнными и отрицательными находками,
- риск-группу,
- оценку кровопотери,
- вероятность сосудистого вмешательства,
- рекомендации по уровню готовности.

Пример:

```json
{
  "schema_version": "1.0",
  "case_info": {
    "gestational_week": 34,
    "previous_cs_count": 2
  },
  "extracted_features": {
    "invasion": {
      "type": "increta",
      "confidence": "probable"
    },
    "anatomy": {
      "bladder_involvement": "possible",
      "parametrium_involvement": "absent",
      "posterior_wall_involvement": "absent"
    },
    "placenta_location": {
      "placenta_previa": "present",
      "anterior_placenta": "present"
    },
    "mri_signs": {
      "retroplacental_vessels": "present",
      "lacunae": "present",
      "uterine_wall_thinning": "present",
      "uterine_hernia_or_bulging": "absent"
    },
    "clinical_context": {
      "preoperative_bleeding": "absent"
    }
  },
  "evidence": {
    "positive_findings": [],
    "uncertain_findings": [],
    "negative_findings": []
  },
  "score": {
    "clinical_score": 9,
    "risk_group": "moderate",
    "red_flag": 0,
    "score_reasons": "increta: +3; вероятное/возможное врастание: +1"
  },
  "predicted_risks": {
    "massive_blood_loss_over_1500_ml_percent": 35,
    "estimated_blood_loss_ml_range": "1000–1500 мл",
    "vascular_intervention_percent": 25,
    "bladder_involvement_percent": 15,
    "risk_summary_text": "Риск массивной кровопотери >1500 мл: 35%; прогнозируемый объём кровопотери: 1000–1500 мл"
  },
  "recommendation": {
    "readiness_level": "2",
    "readiness_text": "Уровень 2: умеренный риск, усиленная подготовка."
  },
  "computed_rationale": "Уровень готовности выбран на основании признаков: increta."
}
```

---

# Ограничения текущей версии

- Маленькие модели иногда генерируют нестабильный JSON
- Качество evidence зависит от размера модели
- Rule-based extraction реализован как baseline/fallback
- Пока поддерживается только single-case inference

---

# Планируемые улучшения

- Hybrid extraction (LLM + rules)
- Batch processing
- Evaluation pipeline
- Более крупные medical LLM
- Docker
- API / web-интерфейс
