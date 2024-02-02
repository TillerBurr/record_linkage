import re
from copy import copy
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import polars as pl
from nameparser import HumanName
from pydantic import BaseModel
from rich import print
from scourgify import NormalizeAddress
from scourgify.exceptions import UnParseableAddressError
from tenacity import retry, stop_after_attempt

from record_linkage.utils import then_replace

FIRST_PERSON_FIELDS = ["title1", "first1", "middle1", "last1", "nickname1", "suffix1"]
SECOND_PERSON_FIELDS = ["title2", "first2", "middle2", "last2", "nickname2", "suffix2"]
PERSON_FIELDS = ["title", "first", "middle", "last", "nickname", "suffix"]
FIRST_RENAME_DICT = {y: x for x, y in zip(PERSON_FIELDS, FIRST_PERSON_FIELDS)}
SECOND_RENAME_DICT = {y: x for x, y in zip(PERSON_FIELDS, SECOND_PERSON_FIELDS)}
ADDRESS_FIELDS = [
    "address_line_1",
    "address_line_2",
    "city",
    "state",
    "postal_code",
]

COMMON_FIELDS = PERSON_FIELDS + ADDRESS_FIELDS
ID_COLUMN = "ID"

# regular expressions to find po box and zip code
zipexpr = re.compile(r"\d{5}(-?\d{4})?$")
boxexpr = re.compile(r"^P\.?O\.? box (\d+)", flags=re.IGNORECASE)


class AggColCastError(Exception):
    """Raised when the aggregate column cannot be converted to a Float64."""


class NormalizeConfig(BaseModel):
    """Normalization Configuration
    Attributes:
        addr_col: The name of the Address column. Should be a full mailing address.
            (US only, due to usaddress-scourgify requirement)
        name_col: The name of the Name column. Should be a person's or a couple's
            name. If a couple exists, it should be separated by a '&' or ' and '.
        agg_col: The name of the column that will be aggregated. The column either
            should be a number that can be aggregated by polars, or a string
            representing a number, e.g. a dollar amount "$23.32".
        id_column: The name of the column that will be used as a unique ID. The only
            requirement on this column is that the values in it are unique.


    """

    addr_col: str | None = None
    name_col: str | None = None
    agg_col: str | None = None
    addl_cols_to_keep: str | list[str] | None = None
    id_column: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.id_column is None:
            self.id_column = ID_COLUMN
        print(self)


class Normalize:
    def __init__(self, df_in: pl.DataFrame, config: NormalizeConfig) -> None:
        """Normalize object is a wrapper around a DataFrame that normalizes a DataFrame
        containing names and addresses.

        Args:
            df_in: DataFrame with a name column and and address column, as specified in
                the config.
            config: Configuration for normalization.
        """
        self.config = config
        self.id_column = (
            self.config.id_column if self.config.id_column is not None else ID_COLUMN
        )
        df_in = self.add_id(df_in)
        df_in = self.fill_null_ids(df_in, self.id_column)
        self.df = df_in
        self.original_df = copy(df_in)

    def normalize(self) -> pl.DataFrame:
        if self.config.agg_col is not None:
            self.fix_agg_col()
            self.aggregate_df()

        self.normalize_name()
        self.normalize_address()
        self._one_person_per_row()
        self.lowercase_strings()
        return self.extract_final_dataframe()

    def fix_agg_col(self) -> None:
        """Transform the `agg_col` into a `pl.Float64` if it is a string"""

        if (
            self.df.select(self.config.agg_col).dtypes[0] == pl.Utf8
            and self.config.agg_col
        ):
            try:
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
            except pl.ComputeError:
                raise AggColCastError(
                    f"Unable to cast {self.config.agg_col} as a Float",
                )

    @staticmethod
    def fill_null_ids(df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Fill a column with unique values if it contains null values.

        Args:
            df: DataFrame with `column` as a column.
            column: Column name to fill with UUID values if null.

        Returns: DataFrame with the `column` filled with unique values. The `column`
            will now have the datatype `pl.Utf8`.

        """
        if df.select(column).dtypes[0] != pl.Utf8:
            df = df.with_columns(pl.col(column).cast(pl.Utf8))

        nulls = df.get_column(column).is_null().sum()

        if nulls > 0:
            null_expr = pl.col(column).is_null()
            null_mask = df.select(null_expr).get_column(column)
            # minimize calls to `uuid4`, only call when needed
            _unique_ = [str(uuid4()) if x else None for x in null_mask]
            df = df.with_columns(_unique_=pl.Series(_unique_))

            when = pl.when(null_expr)
            df = df.with_columns(then_replace(when, pl.col("_unique_"), column))
            df = df.drop("_unique_")
        return df

    def aggregate_df(self) -> Self:
        """Aggregate multiple rows into one row by grouping by `id_column` and summing
        the `agg_col`
        """
        if self.config.agg_col is None:
            raise ValueError("`agg_col` Cannot be `None`.")
        agg_df = self.df.groupby(self.id_column).agg(
            pl.col(self.config.agg_col).sum(),
        )
        self.drop_dupes()
        df = self.df.drop(self.config.agg_col)
        self.df = df.join(agg_df, on=self.id_column)
        return self

    def drop_dupes(self) -> Self:
        """Drop duplicate rows based on `id_column`."""

        self.df = self.df.unique(subset=self.id_column)
        return self

    def add_id(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add the `id_column` if it does not exist in the DataFrame. The new
        `id_column` is a column full of nulls.

        Args:
            df: DataFrame to add an id column, if it doesn't exist.

        Returns: DataFrame with an id column.

        """

        if self.id_column not in df.schema:
            df = df.with_columns(pl.lit(None).alias(self.id_column))
        return df

    def normalize_name(self) -> Self:
        """Normalize the `name_col` from the `config` object."""

        print("[green]Normalizing Name")
        if self.config.name_col is not None:
            df = self.df.with_columns(
                pl.col(self.config.name_col)
                .apply(_normalize_name_str)
                .alias("std_name"),
            ).unnest("std_name")
            self.df = replace_empty_strings(
                df,
            )
        else:
            self.df = self.df.with_columns(
                pl.lit(None).alias(field) for field in PERSON_FIELDS
            )

        return self

    @retry(stop=stop_after_attempt(3))
    def normalize_address(self) -> Self:
        """Normalize the `addr_col` from the `config` object. Retry up to 3 times. It
        is possible for the normalization to fail, since `usaddress` uses probabilistic
        models.
        """

        print("[green]Normalizing Address")
        if self.config.addr_col is not None:
            self.df = self.df.with_columns(
                pl.col(self.config.addr_col)
                .apply(_normalize_address_str)
                .alias("std_addr"),
            ).unnest("std_addr")

        else:
            self.df = self.df.with_columns(
                pl.lit(None).alias(field) for field in ADDRESS_FIELDS
            )
        return self

    def extract_final_dataframe(self) -> pl.DataFrame:
        """Extract the final DataFrame as a subset of `self.df`.
        Only returns fields that are necessary for the pipeline or are requested
        explicitly.
        """

        addl = [self.id_column]
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
        """Export the DataFrame to a csv file.

        Args:
            path: Path to write the csv.
        """
        self.df.write_csv(path)

    def _one_person_per_row(self) -> Self:
        """Only allow one person per row.
        In the case of two people in the name column, move the second person to a new
        row.
        """
        if self.config.name_col is None:
            return self

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
        """Make all strings in the DataFrame lowercase.
        This helps the record linkage algorithm to link records properly.
        """

        self.df = self.df.with_columns(pl.col(pl.Utf8).str.to_lowercase())
        return self


def replace_empty_strings(
    df: pl.DataFrame,
    cols: pl.Utf8 | list[str] = pl.Utf8,  # type:ignore
) -> pl.DataFrame:
    """Replace empty strings in given columns with `None`.

    Args:
        df: DataFrame containing `cols`
        cols: Columns to replace empty strings.

    Returns:

    """
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
    """Normalize a name string.
    Split the name into multiple columns if separated by '&' or ' and '. Then
    parse each of these names into their parts using `nameparser`.

    Args:
        name: A name string.

    Returns: Dictionary of name parts for 1-2 people.

    """
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
    """Normalize the address.
    Using `usaddress-scourgify` to parse an address into its components. It tends to
    fail if it is a post office box, in which case, it is handled manually using
    regular expressions.

    Args:
        address:

    Returns:

    """
    try:
        result = NormalizeAddress(address).normalize()
    except UnParseableAddressError:
        if "PO BOX" in address.upper().replace(".", ""):
            pobox = re.search(boxexpr, address)
            pobox = pobox[0] if pobox is not None else ""
            zipcode = re.search(zipexpr, address)
            zipcode = zipcode[0] if zipcode is not None else ""

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
    if result["postal_code"] is not None:
        result["postal_code"] = result["postal_code"][0:5]  # type: ignore
    return result
