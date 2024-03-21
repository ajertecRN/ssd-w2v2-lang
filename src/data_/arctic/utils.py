import re
from typing import Iterable, Optional, Tuple

import pandas as pd

ARCTIC_PROMPT_ROW_REGEX = re.compile(r"\(\s*(?P<id>\S+)\s*\"(?P<text>[^\"]*)\"\s*\)")


def arctic_prompt_line_to_row(line: str) -> Optional[Tuple[str, str, str]]:
    line = str(line).strip()
    matched_row = ARCTIC_PROMPT_ROW_REGEX.match(line)

    if matched_row is None:
        return None

    prompt_id = str(matched_row.group("id")).strip()
    prompt_name = f"{prompt_id}.wav"
    prompt_text = str(matched_row.group("text"))

    return (prompt_id, prompt_name, prompt_text)


def arctic_rows_to_df(rows: Iterable[Tuple[str, str, str]]):
    df = (
        pd.DataFrame(rows, columns=["id", "name", "transcript"])
        .astype(
            {
                "id": str,
                "name": str,
                "transcript": str,
            }
        )
        .dropna()
        .drop_duplicates()
        .sort_values(["id", "name", "transcript"])
        .reset_index(drop=True)
    )

    return df


def parse_arctic_prompts(prompts: str):
    rows = list()
    for line in prompts.splitlines():
        row = arctic_prompt_line_to_row(line=line)
        if row:
            rows.append(row)

    return arctic_rows_to_df(rows=rows)
