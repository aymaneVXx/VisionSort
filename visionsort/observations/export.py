from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def jsonl_to_parquet(*, jsonl_path: Path, parquet_path: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas requis pour exporter en Parquet.") from exc
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow requis pour exporter en Parquet.") from exc

    rows: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    df = pd.json_normalize(rows) if rows else pd.DataFrame()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return {"rows": int(len(df)), "columns": int(len(df.columns)), "parquet_path": str(parquet_path)}

