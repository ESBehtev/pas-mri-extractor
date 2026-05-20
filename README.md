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

---

## Запуск только через regex-правила

```bash
python run_single.py --use-rules --text-file examples/sample_mri.txt
```

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
Qwen/Qwen2.5-1.5B-Instruct
```

Можно легко добавить другие модели:
- Qwen
- MedGemma
- Llama
- локальные HF-модели

---

# Что возвращает пайплайн

Результат содержит:

- извлечённые MRI-признаки,
- clinical summary,
- clinical rationale,
- риск-группу,
- оценку кровопотери,
- вероятность сосудистого вмешательства,
- рекомендации по уровню готовности.

Пример:

```json
{
  "features": {
    "invasion_type": "increta",
    "placenta_previa": "present"
  },
  "score": {
    "risk_group": "high"
  }
}
```

---

# Ограничения текущей версии

- Маленькие модели иногда генерируют нестабильный JSON
- Качество clinical_summary зависит от размера модели
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
