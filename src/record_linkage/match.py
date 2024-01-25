from datetime import timedelta
from pathlib import Path
from time import time
from typing import Self

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
        results_id_column: str,
        save_dir: Path = Path("data/"),
        lower_limit_prob: float = 0.75,
    ) -> None:
        self.results_config = results_config
        self.starting_list_config = starting_list_config
        self.df_results_orig = pl.read_csv(
            results_file_path,
            infer_schema_length=10000,
            dtypes={results_id_column: pl.Utf8},
        )
        self.results_amount_col = (
            results_config.agg_col if results_config.agg_col else "Amount"
        )
        self.df_starting_in = pl.read_csv(
            starting_list_path,
            infer_schema_length=10000,
            dtypes={results_id_column: pl.Utf8},
        )
        self.results_id_column = results_id_column
        self.save_dir = save_dir
        self.lower_limit_prob = lower_limit_prob
        self.is_normalized = False
        self.is_linked = False

    def normalize(self) -> Self:
        self.results = Normalize(self.df_results_orig, self.results_config)
        self.aggregated_results_orig = (
            self.df_results_orig.select(
                pl.col(self.results_id_column),
                pl.col(self.results_amount_col),
            )
            .groupby(self.results_id_column)
            .agg(pl.col(self.results_amount_col).sum())
        )
        self.mailing_list = Normalize(self.df_starting_in, self.starting_list_config)
        self.results_df_for_linking = self.results.extract_final_dataframe()
        self.starting_df_for_linking = self.mailing_list.extract_final_dataframe()
        self.is_normalized = True
        return self

    def link(self) -> Self:
        if not self.is_normalized:
            raise NotNormalizedError
        self.linker = Linker(
            starting_list_df=self.starting_df_for_linking,
            id_column=self.results_id_column,
            results_df=self.results_df_for_linking,
            lower_limit_probability=self.lower_limit_prob,
        )
        print("Estimating Model")
        self.linker.estimate_model()
        print("Predicting Model")

        self.linker.predict()
        # self.linker.save_model(self.save_dir / "model.json")
        self.linker_predictions = pl.from_pandas(
            self.linker.predictions.as_pandas_dataframe(),
        )
        self.is_linked = True
        return self

    def append_ids(self) -> Self:
        results_df_row_num_and_id = self.results.df.select(
            [self.results_config.row_num_col_name, self.results_id_column],
        )
        starting_df_row_num_and_id = self.mailing_list.df.select(
            [self.starting_list_config.row_num_col_name, self.results_id_column],
        )

        df = self.linker_predictions.join(
            results_df_row_num_and_id,
            left_on=f"{self.results_config.row_num_col_name}_r",
            right_on=self.results_config.row_num_col_name,
            how="left",
        ).rename({self.results_id_column: f"{self.results_id_column}_results"})

        df = df.join(
            starting_df_row_num_and_id,
            left_on=f"{self.starting_list_config.row_num_col_name}_l",
            right_on=self.starting_list_config.row_num_col_name,
            how="left",
        ).rename({self.results_id_column: f"{self.results_id_column}_starting"})

        self.predictions = df
        return self

    def make_matches_unique(self) -> Self:
        """ """
        starting_id_col_str = f"{self.results_id_column}_starting"
        starting_id_col = pl.col(starting_id_col_str)
        results_id_col_str = f"{self.results_id_column}_results"
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

    def match(self) -> Self:
        self.normalize()
        self.link()

        if not self.is_normalized:
            raise NotNormalizedError
        if not self.is_linked:
            raise NotLinkedError
        self.append_ids()

        donations = self.results.df.select(
            self.results_id_column,
            self.results_amount_col,
        )
        donations_tmp = self.aggregated_results_orig.rename(
            {self.results_amount_col: "temp_amt"},
        )

        orig_starting_df_lower_id = self.mailing_list.original_df.with_columns(
            pl.col(self.results_id_column).str.to_lowercase(),
        )
        key_match = orig_starting_df_lower_id.join(donations, on=self.results_id_column)
        matched_ids = key_match.get_column(self.results_id_column)
        self.predictions = self.predictions.filter(
            pl.col(f"{self.results_id_column}_starting").is_in(matched_ids).is_not()
            & pl.col(f"{self.results_id_column}_results").is_in(matched_ids).is_not(),
        )

        self.make_matches_unique()

        pred_df = self.predictions.select(
            "match_probability",
            "match_weight",
            f"{self.results_id_column}_starting",
            f"{self.results_id_column}_results",
        )

        non_key_match = orig_starting_df_lower_id.filter(
            pl.col(self.results_id_column).is_in(matched_ids).is_not(),
        )
        matched = non_key_match.join(
            pred_df,
            left_on=self.results_id_column,
            right_on=f"{self.results_id_column}_starting",
            how="left",
        )
        matched = matched.join(
            donations,
            left_on=f"{self.results_id_column}_results",
            right_on=self.results_id_column,
            how="left",
        )
        key_match = key_match.with_columns(
            [
                pl.col(self.results_id_column).alias(
                    f"{self.results_id_column}_results",
                ),
                pl.lit(1.0).alias("match_probability"),
                pl.lit(100.0).alias("match_weight"),
            ],
        ).select(matched.columns)
        matched = key_match.vstack(matched)
        matched = matched.join(
            donations_tmp,
            on=self.results_id_column,
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
                pl.col(self.results_amount_col),
            ),
            then_replace(
                when_clause,
                pl.col(self.results_id_column),
                pl.col(f"{self.results_id_column}_results"),
            ),
            then_replace(when_clause, pl.lit(1.0), pl.col("match_probability")),
            then_replace(when_clause, pl.lit(100.0), pl.col("match_weight")),
        ).drop("temp_amt")

        self.matched = matched.sort(
            by=[f"{self.results_id_column}_results", self.results_amount_col],
            descending=[False, True],
        ).unique(subset=self.results_id_column)
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
        agg_col="Gf_Amount",
        id_column="ID",
        name_col="Gf_CnBio_Name",
        addr_col="Full Address",
    )
    starting_config = NormalizeConfig(
        id_column="ID",
        addr_col="Full Address",
        name_col="Full Name",
    )
    data = LinkedData(
        results_file_path="data/donations.csv",
        starting_list_path="data/starting.csv",
        results_config=results_config,
        starting_list_config=starting_config,
        results_id_column="ID",
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
