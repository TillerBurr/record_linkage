from pathlib import Path
from typing import Mapping

import polars as pl
import polars.functions.whenthen as whenthen


def then_replace(
    when_clause: whenthen.When,
    then_col: pl.Expr | str,
    otherwise_col: str,
) -> pl.Expr:
    """

    Shorthand for
    `when_clause.then(then_col).otherwise(pl.col(otherwise_col)).alias(otherwise_col)`.
    Useful when needing to use multiple `when` clauses in a single expression.
    Args:
        when_clause: Polars `When` object to use in choosing column values.
        Created via `pl.when`.
        then_col: The column to use when the `when_clause` is true.
        otherwise_col: The string name of the column to replace the values of the
        `then_col` if `when_clause` is true.

    Returns: Column Expressions with values of `otherwise_col` replaced
    with `then_col` if `when_clause` is true.

    """
    if isinstance(then_col, str):
        then_col = pl.col(then_col)
    if not isinstance(otherwise_col, str):
        raise TypeError("otherwise_col must be a string")
    rv = (
        when_clause.then(then_col).otherwise(pl.col(otherwise_col)).alias(otherwise_col)
    )
    return rv


def remove_none_kwargs(**kwargs):
    """Remove the keys with None values from kwargs."""
    return {k: v for k, v in kwargs.items() if v is not None}


def check_file_header(
    file_path: Path | str,
    contains: Mapping[str, str | None],
) -> dict[str, bool]:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    found = {k: False for k in contains.keys()}
    cols = pl.read_csv(file_path, infer_schema_length=10000).columns
    print(cols)
    for k, v in contains.items():
        if v in cols and v is not None:
            found[k] = True
    return found
