from pathlib import Path
from typing import Self

import polars as pl
import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
from duckdb import DuckDBPyConnection
from rich import print
from splink.duckdb.linker import DuckDBLinker

"""
Blocking Rules:
    Used to generate estimates for m in the final model
Deterministic Rules:
    Used to generate a starting estimate for λ, the probability two random
    records match.
Comparisons:
    Used to classify/match records.

"""

blocking_rules = [
    "l.first = r.first and substr(l.last,1) = substr(r.last,1)",
    "l.last = r.last",
    "l.address_line_1 = r.address_line_1",
    "substr(l.last,4) = substr(r.last,4) and substr(l.address_line_1,5) = substr(r.address_line_1,5)",  # noqa
    "substr(l.last,4) = substr(r.last,4) and l.postal_code = r.postal_code",
]
deterministic_rules = [
    "l.first = r.first and l.last = r.last and l.address_line_1 = r.address_line_1",
    "l.last = r.last and levenshtein(r.first, l.first) <= 2 and levenshtein(l.address_line_1,r.address_line_1)<=4",  # noqa
    "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code",
]

settings = {
    "link_type": "link_only",
    "blocking_rules_to_generate_predictions": blocking_rules,
    "comparisons": [
        ctl.name_comparison("first", levenshtein_thresholds=2),
        ctl.name_comparison("last", levenshtein_thresholds=[2, 3, 5]),
        cl.exact_match("city", term_frequency_adjustments=True),
        cl.exact_match("state", term_frequency_adjustments=True),
        cl.levenshtein_at_thresholds("address_line_1", 3),
        cl.levenshtein_at_thresholds("address_line_2", 2),
    ],
}


class DBLinker(DuckDBLinker):
    def __init__(
        self,
        input_table_or_tables: list[pl.DataFrame],
        settings_dict: dict | str | None = None,
        connection: str | DuckDBPyConnection = ":memory:",
        set_up_basic_logging: bool = True,
        output_schema: str | None = None,
        input_table_aliases: str | list | None = None,
    ):
        input_tables = [df.to_arrow() for df in input_table_or_tables]
        super().__init__(
            input_tables,
            settings_dict,  # type:ignore
            connection,
            set_up_basic_logging,
            output_schema,  # type:ignore
            input_table_aliases,  # type:ignore
        )

    EM_estimation = DuckDBLinker.estimate_parameters_using_expectation_maximisation


class Linker:
    def __init__(
        self,
        starting_list_df: pl.DataFrame,
        results_df: pl.DataFrame,
        id_column: str,
        debug_mode: bool = False,
        m_estimatation_type: str = "label",
        lower_limit_probability: float = 0.75,
    ):
        self.linker = DBLinker(
            [starting_list_df, results_df],
            settings,
            input_table_aliases=["starting_list", "results"],
        )
        self.id_column = id_column
        self.linker.debug_mode = debug_mode
        self.m_estimation_type = m_estimatation_type
        self.lower_limit_prob = lower_limit_probability

    def estimate_model(self) -> Self:
        for x in blocking_rules:
            print(
                f"[blue]{x}::::{self.linker.count_num_comparisons_from_blocking_rule(x)}",
            )

        print("[bold green]Estimating λ")
        comps = self.linker.cumulative_comparisons_from_blocking_rules_records(
            blocking_rules=blocking_rules,
        )
        print(f"Cumulative Comparisons:::{comps}")
        self.linker.cumulative_num_comparisons_from_blocking_rules_chart(
            blocking_rules,
        ).save("data/comps.png")
        self.linker.estimate_probability_two_random_records_match(
            deterministic_rules,
            recall=0.9,
        )

        print("[bold green]Estimating u")
        self.linker.estimate_u_using_random_sampling(max_pairs=10_000_000, seed=1)
        print("[bold green]Estimating m")
        if self.m_estimation_type == "label":
            self.linker.estimate_m_from_label_column(self.id_column)
        elif self.m_estimation_type == "EM":
            for rule in blocking_rules:
                self.linker.EM_estimation(rule)
        return self

    def predict(self) -> Self:
        self.predictions = self.linker.predict(
            threshold_match_probability=self.lower_limit_prob,
        )
        return self

    def save_model(self, save_path: str | Path) -> None:
        if isinstance(save_path, Path):
            save_path = save_path.as_posix()
        self.linker.save_model_to_json(save_path, overwrite=True)
