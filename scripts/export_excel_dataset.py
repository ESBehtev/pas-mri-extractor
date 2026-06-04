"""
Export the third sheet of the clinical Excel workbook to CSV and JSONL.

The script intentionally does not modify the source workbook. By default it
looks for exactly one .xlsx file in the project root and writes working
artifacts under data/evaluation/, which is ignored by git.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_CSV_NAME = "dataset_sheet3.csv"
DEFAULT_JSONL_NAME = "dataset_sheet3.jsonl"
DEFAULT_SHEET_INDEX = 2

GOLD_FIELDS = [
    "gold_invasion_type",
    "gold_invasion_confidence",
    "gold_blood_loss_ml",
    "gold_blood_loss_class",
    "gold_massive_blood_loss",
    "gold_bladder_involvement",
    "gold_parametrium_involvement",
    "gold_posterior_wall_involvement",
    "gold_placenta_previa",
    "gold_anterior_placenta",
    "gold_retroplacental_vessels",
    "gold_lacunae",
    "gold_uterine_wall_thinning",
    "gold_uterine_hernia_or_bulging",
    "gold_preoperative_bleeding",
    "gold_highest_suspected_extent",
    "gold_percreta_suspicion",
    "gold_bladder_serosa_suspicion",
    "gold_vascular_intervention",
    "gold_pas_type",
    "gold_readiness_level",
    "gold_risk_group",
    "gold_confidence",
    "gold_rationale",
]

CASE_ID_CANDIDATES = [
    "case_id",
    "id",
    "case",
    "номер",
    "номер_случая",
    "№",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the third sheet of a PAS clinical workbook.",
    )
    parser.add_argument(
        "--xlsx",
        default=None,
        help="Path to the source .xlsx workbook. Defaults to auto-detect in root.",
    )
    parser.add_argument(
        "--sheet-index",
        type=int,
        default=DEFAULT_SHEET_INDEX,
        help="Zero-based Excel sheet index. Default: 2 (third sheet).",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Optional Excel sheet name. Overrides --sheet-index.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for exported CSV/JSONL files.",
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
        help="Output CSV file name.",
    )
    parser.add_argument(
        "--jsonl-name",
        default=DEFAULT_JSONL_NAME,
        help="Output JSONL file name.",
    )
    return parser.parse_args()


def find_root_xlsx() -> Path:
    candidates = sorted(
        path
        for path in PROJECT_ROOT.glob("*.xlsx")
        if not path.name.startswith("~$")
    )

    if not candidates:
        raise FileNotFoundError("No .xlsx file found in the project root.")

    if len(candidates) > 1:
        names = "\n".join(f"- {path.name}" for path in candidates)
        raise ValueError(
            "Multiple .xlsx files found in the project root. "
            "Pass --xlsx explicitly.\n"
            f"{names}"
        )

    return candidates[0]


def resolve_xlsx_path(path_value: str | None) -> Path:
    if not path_value:
        return find_root_xlsx()

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Expected an .xlsx file, got: {path}")

    return path.resolve()


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized_map = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized]

    return None


def stable_case_id(index: int) -> str:
    return f"case_{index + 1:06d}"


def ensure_case_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    case_column = find_column(list(dataframe.columns), CASE_ID_CANDIDATES)

    if case_column is None:
        dataframe.insert(
            0,
            "case_id",
            [stable_case_id(index) for index in range(len(dataframe))],
        )
        return dataframe

    values = []
    for index, value in enumerate(dataframe[case_column].tolist()):
        text = "" if pd.isna(value) else str(value).strip()
        values.append(text or stable_case_id(index))

    if case_column == "case_id":
        dataframe[case_column] = values
        return dataframe

    dataframe.insert(0, "case_id", values)
    return dataframe


def ensure_gold_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    existing = {normalize_column_name(column) for column in dataframe.columns}

    for field in GOLD_FIELDS:
        if normalize_column_name(field) not in existing:
            dataframe[field] = ""
            existing.add(normalize_column_name(field))

    return dataframe


def json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return value


def write_jsonl(dataframe: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in dataframe.to_dict(orient="records"):
            safe_record = {
                str(key): json_safe_value(value)
                for key, value in record.items()
            }
            file.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Reading .xlsx files requires openpyxl. Install dependencies in the "
            "active virtual environment:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  pip install openpyxl"
        ) from error


def main() -> None:
    args = parse_args()
    xlsx_path = resolve_xlsx_path(args.xlsx)
    sheet_name: str | int = args.sheet_name if args.sheet_name else args.sheet_index

    require_openpyxl()
    dataframe = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        engine="openpyxl",
        dtype=object,
        keep_default_na=False,
    )
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    dataframe = ensure_case_id(dataframe)
    dataframe = ensure_gold_fields(dataframe)

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / args.csv_name
    jsonl_path = output_dir / args.jsonl_name

    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    write_jsonl(dataframe, jsonl_path)

    summary = {
        "source_xlsx": str(xlsx_path),
        "sheet": sheet_name,
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "csv_path": str(csv_path),
        "jsonl_path": str(jsonl_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
