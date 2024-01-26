from datetime import timedelta
from pathlib import Path
from time import time
from typing import Any, Self

import polars as pl
from polars.type_aliases import UniqueKeepStrategy
from rich import print

from record_linkage.linking import Linker
from record_linkage.normalize import Normalize, NormalizeConfig
from record_linkage.utils import then_replace


class NotNormalizedError(AttributeError):
    ...


class NotLinkedError(AttributeError):
    ...


class LinkedData:
    def __init__(
        self,
        results_file_path: str | Path,
        starting_list_path: str | Path,
        results_config: NormalizeConfig,
        starting_list_config: NormalizeConfig,
        id_column: str,
        save_dir: Path = Path("data/"),
        lower_limit_prob: float = 0.75,
    ) -> None:
        self.results_config = results_config
        self.starting_list_config = starting_list_config
        self.df_results_orig = pl.read_csv(
            results_file_path,
            infer_schema_length=10000,
            dtypes={id_column: pl.Utf8},
        )
        self.results_amount_col = (
            results_config.agg_col if results_config.agg_col else "Amount"
        )
        self.df_starting_in = pl.read_csv(
            starting_list_path,
            infer_schema_length=10000,
            dtypes={id_column: pl.Utf8},
        )
        self.id_column = id_column
        self.save_dir = save_dir
        self.lower_limit_prob = lower_limit_prob
        self.is_normalized = False
        self.is_linked = False

    def normalize(self) -> Self:
        self.results = Normalize(self.df_results_orig, self.results_config)
        self.results.normalize()
        results_orig = self.results.drop_dupes().df
        self.aggregated_results_orig = (
            results_orig.select(
                pl.col(self.id_column),
                pl.col(self.results_amount_col),
            )
            .groupby(self.id_column)
            .agg(pl.col(self.results_amount_col).sum())
        )
        self.starting_list = Normalize(self.df_starting_in, self.starting_list_config)
        self.starting_list.normalize()
        self.is_normalized = True
        self.starting_normalized = self.starting_list.extract_final_dataframe()
        self.results_normalized = self.results.extract_final_dataframe()
        return self

    def link(self, settings: dict[str, Any] = {}) -> Self:
        if not self.is_normalized:
            raise NotNormalizedError
        self.linker = Linker(
            starting_list_df=self.starting_normalized,
            results_df=self.results_normalized,
            lower_limit_probability=self.lower_limit_prob,
            settings=settings,
        )
        print("Estimating Model")
        self.linker.estimate_model()

        print("Predicting Model")
        self.linker.predict()
        self.predictions = pl.from_pandas(
            self.linker.predictions.as_pandas_dataframe(),
        )
        self.predictions.write_csv("data/test.csv")
        self.predictions = self.predictions.rename(
            {
                f"{self.id_column}_l": f"{self.id_column}_starting",
                f"{self.id_column}_r": f"{self.id_column}_results",
            },
        )
        self.is_linked = True
        return self

    def make_matches_unique(self) -> Self:
        """ """
        starting_id_col_str = f"{self.id_column}_starting"
        starting_id_col = pl.col(starting_id_col_str)
        results_id_col_str = f"{self.id_column}_results"
        results_id_col = pl.col(results_id_col_str)
        eq = self.predictions.filter(starting_id_col == results_id_col)

        eq = uniquify(eq, starting_id_col_str)

        not_eq = self.predictions.filter(starting_id_col != results_id_col)
        _filter_starting = starting_id_col.is_in(
            eq.select(starting_id_col).to_series(),
        ).is_not()

        _filter_results = results_id_col.is_in(
            eq.select(results_id_col).to_series(),
        ).is_not()

        df = not_eq.filter(_filter_starting)
        df = df.filter(_filter_results)
        df = uniquify(df, starting_id_col_str)
        df = uniquify(df, results_id_col_str)

        eq = eq.vstack(df)
        self.predictions = eq
        return self

    def match(
        self,
        settings: dict[str, Any] = {},
    ) -> Self:
        self.normalize()
        self.link(settings)

        if not self.is_normalized:
            raise NotNormalizedError
        if not self.is_linked:
            raise NotLinkedError

        donations = self.results.df.select(
            self.id_column,
            self.results_amount_col,
        )
        donations_tmp = self.aggregated_results_orig.rename(
            {self.results_amount_col: "temp_amt"},
        )

        orig_starting_df_lower_id = self.starting_list.original_df.with_columns(
            pl.col(self.id_column).str.to_lowercase(),
        )
        key_match = orig_starting_df_lower_id.join(donations, on=self.id_column)
        matched_ids = key_match.get_column(self.id_column)
        self.predictions = self.predictions.filter(
            pl.col(f"{self.id_column}_starting").is_in(matched_ids).is_not()
            & pl.col(f"{self.id_column}_results").is_in(matched_ids).is_not(),
        )

        self.make_matches_unique()

        pred_df = self.predictions.select(
            "match_probability",
            "match_weight",
            f"{self.id_column}_starting",
            f"{self.id_column}_results",
        )
        non_key_match = orig_starting_df_lower_id.filter(
            pl.col(self.id_column).is_in(matched_ids).is_not(),
        )
        matched = non_key_match.join(
            pred_df,
            left_on=self.id_column,
            right_on=f"{self.id_column}_starting",
            how="left",
        )
        matched = matched.join(
            donations,
            left_on=f"{self.id_column}_results",
            right_on=self.id_column,
            how="left",
        )
        key_match = key_match.with_columns(
            [
                pl.col(self.id_column).alias(
                    f"{self.id_column}_results",
                ),
                pl.lit(1.0).alias("match_probability"),
                pl.lit(100.0).alias("match_weight"),
            ],
        ).select(matched.columns)
        matched = key_match.vstack(matched)
        matched = matched.join(
            donations_tmp,
            on=self.id_column,
            how="left",
        )

        # Check that all keys are matched properly
        when_clause = pl.when(
            pl.col(self.results_amount_col).is_null()
            & pl.col("temp_amt").is_not_null(),
        )
        matched = matched.with_columns(
            then_replace(
                when_clause,
                pl.col("temp_amt"),
                self.results_amount_col,
            ),
            then_replace(
                when_clause,
                pl.col(self.id_column),
                f"{self.id_column}_results",
            ),
            then_replace(when_clause, pl.lit(1.0), "match_probability"),
            then_replace(when_clause, pl.lit(100.0), "match_weight"),
        ).drop("temp_amt")

        self.matched = matched.sort(
            by=[f"{self.id_column}_results", self.results_amount_col],
            descending=[False, True],
        ).unique(subset=self.id_column)
        return self


def uniquify(
    df: pl.DataFrame,
    column: str,
    secondary_sort_col: str = "match_weight",
    keep: UniqueKeepStrategy = "first",
) -> pl.DataFrame:
    unique_df = df.sort(
        by=[column, secondary_sort_col],
        descending=[False, True],
    ).unique(
        subset=[column],
        keep=keep,
    )

    return unique_df


if __name__ == "__main__":
    start = time()
    results_config = NormalizeConfig(
        agg_col="Amount",
        id_column="ID",
        name_col="Name",
        addr_col="Full Address",
    )
    starting_config = NormalizeConfig(
        id_column="ID",
        addr_col="Full Address",
        name_col="Full Name",
    )
    data = LinkedData(
        results_file_path="data/results.csv",
        starting_list_path="data/starting.csv",
        results_config=results_config,
        starting_list_config=starting_config,
        id_column="ID",
        lower_limit_prob=0.6,
    )
    data.match()

    data.predictions.write_csv("data/predicted.csv")
    data.matched.write_csv("data/matched.csv")

    run_time = time() - start
    run_time = str(timedelta(seconds=run_time)).split(":")
    print(
        f"Process took [bold cyan]{run_time[0]} Hours {run_time[1]} Minutes {round(float(run_time[2]),2)} Seconds[/bold cyan]",  # noqa
    )
