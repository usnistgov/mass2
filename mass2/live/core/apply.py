"""Apply each channel's saved mass2 recipe to the new raw records of many channels at once."""

import polars as pl

from mass2.core.misc import PulseDataFromNumpy
from mass2.core.recipe import Recipe


def apply_recipes(recipes: dict[int, Recipe], raw: pl.DataFrame) -> pl.DataFrame:
    """Each channel's rows of `raw` through that channel's mass2 `Recipe.calc_from_df` (what `Channel.with_steps`
    runs), with `good` from the recipe's last good-pulse expression. Rows keep their order; `pulse` is dropped. A
    channel with no recipe passes through with `good = False`."""
    raw = raw.with_row_index("_row")
    parts = []
    for (ch_num,), rows in raw.partition_by("ch_num", as_dict=True, maintain_order=True).items():
        recipe = recipes.get(ch_num)
        if recipe is None:
            parts.append(rows.drop("pulse").with_columns(good=pl.lit(False)))
        else:
            analyzed = recipe.calc_from_df(rows.drop("pulse"), PulseDataFromNumpy(rows["pulse"].to_numpy()))
            parts.append(analyzed.with_columns(good=recipe[-1].good_expr.fill_null(False)))
    return pl.concat(parts, how="diagonal_relaxed").sort("_row").drop("_row")
