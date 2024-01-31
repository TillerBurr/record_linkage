from typing import Any, Sequence, TypeAlias

import splink.duckdb.comparison_library as cl
import splink.duckdb.comparison_template_library as ctl
from splink.comparison import Comparison

T_SETTINGS: TypeAlias = dict[str, Sequence[str | Comparison]]

"""
Blocking Rules:
    Used to generate estimates for m in the final model
Deterministic Rules:
    Used to generate a starting estimate for λ, the probability two random
    records match.
Comparisons:
    Used to classify/match records.

"""
# TODO: Use the json/dict version of the settings to create defaults.
BASE_SETTINGS = {
    "unique_id_column_name": "unique_id",
    "blocking_rules_to_generate_predictions": [
        {
            "blocking_rule": "l.first = r.first and substr(l.last,1) = substr(r.last,1)",  # noqa
            "sql_dialect": None,
        },
        {"blocking_rule": "l.last = r.last", "sql_dialect": None},
        {"blocking_rule": "l.address_line_1 = r.address_line_1", "sql_dialect": None},
        {
            "blocking_rule": "substr(l.last,4) = substr(r.last,4) and substr(l.address_line_1,5) = substr(r.address_line_1,5)",  # noqa
            "sql_dialect": None,
        },
        {
            "blocking_rule": "substr(l.last,4) = substr(r.last,4) and l.postal_code = r.postal_code",  # noqa
            "sql_dialect": None,
        },
    ],
    "comparisons": [
        {
            "output_column_name": "first",
            "comparison_levels": [
                {
                    "sql_condition": '"first_l" IS NULL OR "first_r" IS NULL',
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"first_l" = "first_r"',
                    "label_for_charts": "Exact match first",
                },
                {
                    "sql_condition": 'levenshtein("first_l", "first_r") <= 2',
                    "label_for_charts": "Levenshtein <= 2",
                },
                {
                    "sql_condition": 'damerau_levenshtein("first_l", "first_r") <= 1',
                    "label_for_charts": "Damerau_levenshtein <= 1",
                },
                {
                    "sql_condition": 'jaro_winkler_similarity("first_l", "first_r") >= 0.9',  # noqa
                    "label_for_charts": "Jaro_winkler_similarity >= 0.9",
                },
                {
                    "sql_condition": 'jaro_winkler_similarity("first_l", "first_r") >= 0.8',  # noqa
                    "label_for_charts": "Jaro_winkler_similarity >= 0.8",
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. First within levenshtein threshold 1 vs. First within damerau-levenshtein threshold 1 vs. First within jaro_winkler thresholds 0.9, 0.8 vs. anything else",  # noqa
        },
        {
            "output_column_name": "last",
            "comparison_levels": [
                {
                    "sql_condition": '"last_l" IS NULL OR "last_r" IS NULL',
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"last_l" = "last_r"',
                    "label_for_charts": "Exact match last",
                },
                {
                    "sql_condition": 'levenshtein("last_l", "last_r") <= 2',
                    "label_for_charts": "Levenshtein <= 2",
                },
                {
                    "sql_condition": 'levenshtein("last_l", "last_r") <= 3',
                    "label_for_charts": "Levenshtein <= 3",
                },
                {
                    "sql_condition": 'levenshtein("last_l", "last_r") <= 5',
                    "label_for_charts": "Levenshtein <= 5",
                },
                {
                    "sql_condition": 'damerau_levenshtein("last_l", "last_r") <= 1',
                    "label_for_charts": "Damerau_levenshtein <= 1",
                },
                {
                    "sql_condition": 'jaro_winkler_similarity("last_l", "last_r") >= 0.9',  # noqa
                    "label_for_charts": "Jaro_winkler_similarity >= 0.9",
                },
                {
                    "sql_condition": 'jaro_winkler_similarity("last_l", "last_r") >= 0.8',  # noqa
                    "label_for_charts": "Jaro_winkler_similarity >= 0.8",
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. Last within levenshtein threshold 1 vs. Last within damerau-levenshtein threshold 1 vs. Last within jaro_winkler thresholds 0.9, 0.8 vs. anything else",  # noqa
        },
        {
            "output_column_name": "city",
            "comparison_levels": [
                {
                    "sql_condition": '"city_l" IS NULL OR "city_r" IS NULL',
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"city_l" = "city_r"',
                    "label_for_charts": "Exact match",
                    "tf_adjustment_column": "city",
                    "tf_adjustment_weight": 1.0,
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. anything else",
        },
        {
            "output_column_name": "state",
            "comparison_levels": [
                {
                    "sql_condition": '"state_l" IS NULL OR "state_r" IS NULL',
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"state_l" = "state_r"',
                    "label_for_charts": "Exact match",
                    "tf_adjustment_column": "state",
                    "tf_adjustment_weight": 1.0,
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. anything else",
        },
        {
            "output_column_name": "address_line_1",
            "comparison_levels": [
                {
                    "sql_condition": '"address_line_1_l" IS NULL OR "address_line_1_r" IS NULL',  # noqa
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"address_line_1_l" = "address_line_1_r"',
                    "label_for_charts": "Exact match",
                },
                {
                    "sql_condition": 'levenshtein("address_line_1_l", "address_line_1_r") <= 3',  # noqa
                    "label_for_charts": "Levenshtein <= 3",
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. Address_Line_1 within levenshtein threshold 3 vs. anything else",  # noqa
        },
        {
            "output_column_name": "address_line_2",
            "comparison_levels": [
                {
                    "sql_condition": '"address_line_2_l" IS NULL OR "address_line_2_r" IS NULL',  # noqa
                    "label_for_charts": "Null",
                    "is_null_level": True,
                },
                {
                    "sql_condition": '"address_line_2_l" = "address_line_2_r"',
                    "label_for_charts": "Exact match",
                },
                {
                    "sql_condition": 'levenshtein("address_line_2_l", "address_line_2_r") <= 2',  # noqa
                    "label_for_charts": "Levenshtein <= 2",
                },
                {"sql_condition": "ELSE", "label_for_charts": "All other comparisons"},
            ],
            "comparison_description": "Exact match vs. Address_Line_2 within levenshtein threshold 2 vs. anything else",  # noqa
        },
    ],
    "link_type": "link_only",
}

DEFAULT_BLOCKING_RULES = (
    "l.first = r.first and substr(l.last,1) = substr(r.last,1)",
    "l.last = r.last",
    "l.address_line_1 = r.address_line_1",
    "substr(l.last,4) = substr(r.last,4) and substr(l.address_line_1,5) = substr(r.address_line_1,5)",  # noqa
    "substr(l.last,4) = substr(r.last,4) and l.postal_code = r.postal_code",
)
DEFAULT_DETERMINISTIC_RULES = (
    "l.first = r.first and l.last = r.last",
    "l.first = r.first and l.last = r.last and l.address_line_1 = r.address_line_1",
    "l.last = r.last and levenshtein(r.first, l.first) <= 2 and levenshtein(l.address_line_1,r.address_line_1)<=4",  # noqa
    "l.address_line_1 = r.address_line_1 and l.postal_code = r.postal_code",
)
DEFAULT_COMPARISONS = (
    ctl.name_comparison("first", levenshtein_thresholds=2),
    ctl.name_comparison("last", levenshtein_thresholds=[2, 3, 5]),
    cl.exact_match("city", term_frequency_adjustments=True),
    cl.exact_match("state", term_frequency_adjustments=True),
    cl.levenshtein_at_thresholds("address_line_1", 3),
    cl.levenshtein_at_thresholds("address_line_2", 2),
)

DEFAULT_UNIQUE_ID_COLUMN_NAME = "unique_id"


class BaseLinkingSettings:
    blocking_rules_to_generate_predictions: list[str] = list(DEFAULT_BLOCKING_RULES)
    comparisons: list[Comparison] = list(DEFAULT_COMPARISONS)
    deterministic_rules: list[str] = list(DEFAULT_DETERMINISTIC_RULES)
    unique_id_column_name: str = DEFAULT_UNIQUE_ID_COLUMN_NAME


def create_settings(settings: dict[str, Any] = {}) -> dict[str, Any]:
    """Create a settings dict that contains the minimal settings.

    Args:
        settings: A dictionary of settings for `splink`.

    Returns: Minimal settings dict to link.

    """
    for setting in (
        "blocking_rules_to_generate_predictions",
        "comparisons",
        "unique_id_column_name",
    ):
        if not settings.get(setting):
            settings[setting] = getattr(BaseLinkingSettings, setting)
    if not settings.get("link_type") or settings.get("link_type") != "link_only":
        settings["link_type"] = "link_only"
    return settings
