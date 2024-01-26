from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

import polars as pl
from duckdb import DuckDBPyConnection
from rich import print
from splink.duckdb.linker import DuckDBLinker

from record_linkage.linking_settings import BaseLinkingSettings, create_settings

T_Estimation: TypeAlias = Literal["label"] | Literal["EM"]


class ModelEstimationError(Exception):
    ...


class DBLinker(DuckDBLinker):
    """Wrapper around splink's `DuckDBLinker`"""

    def __init__(
        self,
        input_table_or_tables: list[pl.DataFrame],
        connection: str | DuckDBPyConnection = ":memory:",
        set_up_basic_logging: bool = True,
        output_schema: str | None = None,
        input_table_aliases: str | list | None = None,
        settings: dict[str, Any] = {},
    ):
        input_tables = [df.to_arrow() for df in input_table_or_tables]

        super().__init__(
            input_tables,
            settings,
            connection,
            set_up_basic_logging,
            output_schema,  # type:ignore
            input_table_aliases,  # type:ignore
        )

    EM_estimation = DuckDBLinker.estimate_parameters_using_expectation_maximisation


class Linker:
    """The Linker. Takes two dataframes and creates a new dataframe with records linked
    together. DataFrames must be normalized before use.

    Attributes:
        settings: Settings for the DuckDBLinker
        linker: The DuckDBLinker
        debug_mode: Debug mode for DuckDBLinker
        m_estimation_type: `m` parameter estimation. By label or by
        Expectation/Maximization
        lower_limit_prob: The lowest probability level acceptible
        predictions: The final predictions/links.
    """

    def __init__(
        self,
        starting_list_df: pl.DataFrame,
        results_df: pl.DataFrame,
        debug_mode: bool = False,
        m_estimatation_type: str = "label",
        lower_limit_probability: float = 0.75,
        settings: dict[str, Any] = {},
    ):
        self.settings = create_settings(settings)
        self.linker = DBLinker(
            [starting_list_df, results_df],
            settings=self.settings,
            input_table_aliases=["a_starting_list", "b_results"],
            # make sure that starting_list_df is the first table
            # see https://github.com/moj-analytical-services/splink/discussions/1660#discussioncomment-7362000 # noqa
        )
        self.linker.debug_mode = debug_mode
        self.m_estimation_type = m_estimatation_type
        self.lower_limit_prob = lower_limit_probability
        self.predictions = None
        self._model_estimated = False

    def estimate_model(
        self,
    ) -> Self:
        """Make model estimates. Needed before making predictions."""
        deterministic_rules = self.settings.get(
            "deterministic_rules",
            BaseLinkingSettings.deterministic_rules,
        )
        blocking_rules = self.settings.get(
            "blocking_rules_to_generate_predictions",
            BaseLinkingSettings.blocking_rules_to_generate_predictions,
        )
        for x in blocking_rules:
            print(
                f"[blue]{x}::::{self.linker.count_num_comparisons_from_blocking_rule(x)}",
            )

        print("[bold green]Estimating λ")
        comps = self.linker.cumulative_comparisons_from_blocking_rules_records(
            blocking_rules=blocking_rules,  # type:ignore
        )
        print(f"Cumulative Comparisons:::[blue]{comps}")
        self.linker.estimate_probability_two_random_records_match(
            deterministic_rules,
            recall=0.9,
        )

        print("[bold green]Estimating u")
        self.linker.estimate_u_using_random_sampling(max_pairs=10_000_000, seed=1)
        print("[bold green]Estimating m")
        if self.m_estimation_type == "label":
            self.linker.estimate_m_from_label_column(
                self.settings.get("unique_id_column_name", "unique_id"),
            )
        elif self.m_estimation_type == "EM":
            for rule in blocking_rules:
                self.linker.EM_estimation(rule)
        self._model_estimated = True
        return self

    def predict(self) -> Self:
        """Make predictions based on the estimates of parameters."""
        if not self._model_estimated:
            raise ModelEstimationError("Model has not yet been estimated")
        self.predictions = self.linker.predict(
            threshold_match_probability=self.lower_limit_prob,
        )
        return self

    def save_model(self, save_path: str | Path) -> None:
        """Save the model for future use.

        Args:
            save_path: Where to save the model.
        """
        if isinstance(save_path, Path):
            save_path = save_path.as_posix()
        self.linker.save_model_to_json(save_path, overwrite=True)

    def load_model(self, load_path: Path) -> None:
        """Load a model from the given path.

        Args:
            load_path: A previously saved model.
        """
        self.linker.load_model(load_path)
        self._model_estimated = True
