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


DEFAULT_BLOCKING_RULES = (
    "l.first = r.first and substr(l.last,1) = substr(r.last,1)",
    "l.last = r.last",
    "l.address_line_1 = r.address_line_1",
    "substr(l.last,4) = substr(r.last,4) and substr(l.address_line_1,5) = substr(r.address_line_1,5)",  # noqa
    "substr(l.last,4) = substr(r.last,4) and l.postal_code = r.postal_code",
)
DEFAULT_DETERMINISTIC_RULES = (
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
