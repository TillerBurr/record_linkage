import re
from copy import copy
from pathlib import Path
from typing import Self
from uuid import uuid4

import polars as pl
from nameparser import HumanName
from pydantic import BaseModel
from rich import print
from scourgify import NormalizeAddress
from scourgify.exceptions import UnParseableAddressError
from tenacity import retry, stop_after_attempt

FIRST_PERSON_FIELDS = ["title1", "first1", "middle1", "last1", "nickname1", "suffix1"]
SECOND_PERSON_FIELDS = ["title2", "first2", "middle2", "last2", "nickname2", "suffix2"]
PERSON_FIELDS = ["title", "first", "middle", "last", "nickname", "suffix"]
FIRST_RENAME_DICT = {y: x for x, y in zip(PERSON_FIELDS, FIRST_PERSON_FIELDS)}
SECOND_RENAME_DICT = {y: x for x, y in zip(PERSON_FIELDS, SECOND_PERSON_FIELDS)}

COMMON_FIELDS = PERSON_FIELDS + [
    "address_line_1",
    "address_line_2",
    "city",
    "state",
    "postal_code",
]

# regular expressions to find po box and zip code
zipexpr = re.compile(r"\d{5}(-?\d{4})?$")
boxexpr = re.compile(r"^P\.?O\.? box (\d+)", flags=re.IGNORECASE)


class NormalizeConfig(BaseModel):
    addr_col: str = "Full Address"
    name_col: str = "Name"
    row_num_col_name: str = "unique_id"
    agg_col: str | None = None
    addl_cols_to_keep: str | list[str] | None = None
    name: str = "Normalized"
    id_column: str = "Account_Number"


class Normalize:
    def __init__(self, df_in: pl.DataFrame, config: NormalizeConfig) -> None:
        self.config = config
        id_column = self.config.id_column
        df_in = self.fill_null_ids(df_in, id_column)
        self.df = df_in
        self.original_df = copy(df_in)
        if self.config.agg_col is not None:
            self.fix_agg_col()
            self.aggregate_df()

        self.normalize_name()
        self.normalize_address()
        self._one_person_per_row()
        self.add_id()
        self.lowercase_strings()

    def fix_agg_col(self) -> None:
        if self.df.select(self.config.agg_col).dtypes[0] == pl.Utf8:
            self.df = self.df.with_columns(
                pl.col(self.config.agg_col)
                .str.strip()
                .str.replace(
                    "[$. ]",
                    "",
                )
                .cast(pl.Float64)
                .alias(self.config.agg_col),
            )

    @staticmethod
    def fill_null_ids(df: pl.DataFrame, column: str) -> pl.DataFrame:
        if df.select(column).dtypes[0] != pl.Utf8:
            df = df.with_columns(pl.col(column).cast(pl.Utf8))
        nulls = df.select(pl.col(column).is_null().sum()).to_series().sum()
        if nulls > 0:
            _unique_ = [str(uuid4()) for _ in range(0, df.shape[0])]
            df = df.with_columns(_unique_=pl.Series(_unique_))
            df = df.with_columns(
                pl.when(pl.col(column).is_null())
                .then(pl.col("_unique_"))
                .otherwise(pl.col(column))
                .alias(column),
            )
            df = df.drop("_unique_")
        return df

    def aggregate_df(self) -> Self:
        if self.config.agg_col is None:
            raise ValueError("`agg_col` Cannot be `None`.")
        agg_df = self.df.groupby(self.config.id_column).agg(
            pl.col(self.config.agg_col).sum(),
        )
        self.drop_dupes()
        df = self.df.drop(self.config.agg_col)
        self.df = df.join(agg_df, on=self.config.id_column)
        return self

    def drop_dupes(self) -> Self:
        self.df = self.df.unique(subset=self.config.id_column)
        return self

    def add_id(self) -> Self:
        if self.config.row_num_col_name not in self.df.schema:
            self.df = self.df.with_row_count(self.config.row_num_col_name)
        return self

    @retry(stop=stop_after_attempt(3))
    def normalize_name(self) -> Self:
        print("[green]Normalizing Name")
        df = self.df.with_columns(
            pl.col(self.config.name_col).apply(_normalize_name_str).alias("std_name"),
        ).unnest("std_name")
        self.df = replace_empty_strings(
            df,
        )

        return self

    @retry(stop=stop_after_attempt(3))
    def normalize_address(self) -> Self:
        print("[green]Normalizing Address")
        self.df = self.df.with_columns(
            pl.col(self.config.addr_col)
            .apply(_normalize_address_str)
            .alias("std_addr"),
        ).unnest("std_addr")
        return self

    def extract_final_dataframe(self) -> pl.DataFrame:
        addl = [self.config.row_num_col_name, self.config.id_column]
        match self.config.addl_cols_to_keep:
            case None:
                pass
            case str():
                addl += [self.config.addl_cols_to_keep]
            case list():
                addl += self.config.addl_cols_to_keep
            case _:
                raise TypeError(
                    f"The type {type(self.config.addl_cols_to_keep)} is not supported",
                )
        cols = COMMON_FIELDS + addl

        self.df_out = self.df.select([pl.col(x) for x in cols])
        return self.df_out

    def export(self, path: str | Path) -> None:
        self.df.write_csv(path)

    def _one_person_per_row(self) -> Self:
        df = self.df

        first = df.select(pl.all().exclude(SECOND_PERSON_FIELDS)).rename(
            FIRST_RENAME_DICT,
        )
        first = first.filter(~pl.all(pl.col(PERSON_FIELDS).is_null()))
        second = df.select(pl.all().exclude(FIRST_PERSON_FIELDS)).rename(
            SECOND_RENAME_DICT,
        )
        second = second.filter(~pl.all(pl.col(PERSON_FIELDS).is_null()))
        if not second.is_empty():
            self.df = first.vstack(second)
        else:
            self.df = first
        return self

    def lowercase_strings(self) -> Self:
        self.df = self.df.with_columns(pl.col(pl.Utf8).str.to_lowercase())
        return self


def replace_empty_strings(
    df: pl.DataFrame,
    cols: pl.Utf8 | list[str] = pl.Utf8,
) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.when(pl.col(cols).str.lengths() == 0)
            .then(None)
            .otherwise(pl.col(cols))
            .keep_name(),
        ],
    )
    return df


def _normalize_name_str(name: str) -> dict[str, str]:
    row = {}
    name = name.replace(" and ", "&")
    name1, *name2 = name.split("&")
    if len(name1.split()) == 1:
        name1 += f" {name2[-1].split()[-1]}"
    iter_list = (name1, name2[-1] if len(name2) > 0 else None)
    for i, person in enumerate(iter_list):
        if person is None:
            row |= {
                f"title{i+1}": None,
                f"first{i+1}": None,
                f"middle{i+1}": None,
                f"last{i+1}": None,
                f"suffix{i+1}": None,
                f"nickname{i+1}": None,
            }
        else:
            person = re.sub(r"[^\w\s]", "", person)
            person = re.sub(r"\d", "", person)
            person = re.sub(" +", " ", person)
            person = person.lower()
            parsed = HumanName(person)
            row |= {
                f"title{i+1}": parsed.title,
                f"first{i+1}": parsed.first,
                f"middle{i+1}": parsed.middle,
                f"last{i+1}": parsed.last,
                f"suffix{i+1}": parsed.suffix,
                f"nickname{i+1}": parsed.nickname,
            }
    return row


def _normalize_address_str(address: str) -> dict[str, str]:
    try:
        result = NormalizeAddress(address).normalize()
    except UnParseableAddressError:
        if "PO BOX" in address.upper().replace(".", ""):
            pobox = re.search(boxexpr, address)[0]
            zipcode = re.search(zipexpr, address)[0]

            citystate = address.replace(pobox, "").replace(zipcode, "").strip()
            split_citystate = citystate.split(",")
            if len(split_citystate) > 3:
                city_idx = 1
                state_idx = 2
            else:
                city_idx = 0
                state_idx = 1
            result = {
                "address_line_1": pobox.upper(),
                "address_line_2": None,
                "city": split_citystate[city_idx].upper().strip(),
                "postal_code": zipcode,
                "state": split_citystate[state_idx].upper().strip(),
            }

        else:
            raise
    result["postal_code"] = result["postal_code"][0:5]
    return result
