"""
Compatibility utilities for transferring data between mass and mass2.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .channel import Channel, ChannelHeader
from .channels import Channels

if TYPE_CHECKING:
    import mass.off  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _leaf_data_sources(header: ChannelHeader) -> tuple[str | None, ...]:
    """Return the raw file path(s) backing a channel's header."""
    leaf_data_sources = getattr(header, "leaf_data_sources", None)
    if leaf_data_sources is not None:
        return leaf_data_sources()
    return (header.data_source,)


def inject_mass_into_mass2(
    mass_group: mass.off.ChannelGroup,
    mass2_channels: Channels,
    columns: Sequence[str],
) -> Channels:
    """
    Copy columns from a mass ChannelGroup into a Channels object.

    Parameters
    ----------
    mass_group : mass.off.ChannelGroup
        mass ChannelGroup with analysis already applied.
    mass2_channels : Channels
        Channels to inject results into.
    columns : Sequence[str]
        Names of mass recipe outputs to copy (e.g. ``["energy", "filtValue"]``).

    Returns
    -------
    Channels
        Updated Channels with new columns appended to each channel's DataFrame.
        All channels in ``mass_group``, including those marked bad, are processed.
        Channels present in ``mass_group`` but absent from ``mass2_channels`` are
        skipped with a warning.

    Raises
    ------
    ValueError
        If pulse counts differ between a mass and mass2 channel.
    """
    from mass.off.util import NoCutInds  # noqa: PLC0415

    updated: dict[int, Channel] = dict(mass2_channels.channels)
    with mass_group.includeBad():
        for ch_num, old_ch in mass_group.items():
            if ch_num not in updated:
                logger.warning("channel %d is in mass but not in mass2 — skipping", ch_num)
                continue
            new_ch = updated[ch_num]
            n_mass = len(old_ch)
            if n_mass != new_ch.npulses:
                raise ValueError(
                    f"channel {ch_num}: mass has {n_mass} pulses but mass2 has {new_ch.npulses}"
                )
            series = [pl.Series(col, old_ch.getAttr(col, NoCutInds())) for col in columns]
            updated[ch_num] = new_ch.with_columns(*series)
    return dataclasses.replace(mass2_channels, channels=updated)


def inject_mass2_into_mass(
    mass2_channels: Channels,
    mass_group: mass.off.ChannelGroup,
    columns: Sequence[str],
) -> None:
    """
    Copy columns from a Channels object into a mass ChannelGroup.

    Parameters
    ----------
    mass2_channels : Channels
        Channels with analysis already applied.
    mass_group : mass.off.ChannelGroup
        mass ChannelGroup to receive the results (mutated in place).
    columns : Sequence[str]
        Names of DataFrame columns to copy.

    Raises
    ------
    ValueError
        If pulse counts differ between a mass2 and mass channel, or if any
        requested column is absent from a channel's DataFrame.
    """
    for ch_num, new_ch in mass2_channels.channels.items():
        if ch_num not in mass_group:
            continue
        missing = [col for col in columns if col not in new_ch.df.columns]
        if missing:
            raise ValueError(
                f"channel {ch_num}: columns {missing} not found in DataFrame. "
                f"Available: {new_ch.df.columns}"
            )
        if new_ch.npulses != len(mass_group[ch_num]):
            raise ValueError(
                f"channel {ch_num}: mass2 has {new_ch.npulses} pulses "
                f"but mass has {len(mass_group[ch_num])}"
            )

    for ch_num, new_ch in mass2_channels.channels.items():
        if ch_num not in mass_group:
            logger.warning("channel %d is in mass2 but not in mass — skipping", ch_num)
            continue
        old_ch = mass_group[ch_num]
        for col in columns:
            arr = new_ch.df[col].to_numpy()
            old_ch.recipes.add(col, lambda arr=arr: arr, ingredients=[], overwrite=True)


def mass_to_mass2(
    mass_group: mass.off.ChannelGroup,
    columns: Sequence[str],
) -> Channels:
    """
    Create a Channels from a mass ChannelGroup and inject analysis results.

    Parameters
    ----------
    mass_group : mass.off.ChannelGroup
        mass ChannelGroup with analysis already applied.
    columns : Sequence[str]
        Names of mass recipe outputs to inject (e.g. ``["energy", "filtValue"]``).

    Returns
    -------
    Channels
        Channels loaded from the same OFF files, with mass results injected.
        All channels including those marked bad in mass are included.
    """
    with mass_group.includeBad():
        off_paths = [ch.offFile.filename for ch in mass_group.values()]
    mass2_channels = Channels.from_off_paths(off_paths, description=mass_group.shortName)
    return inject_mass_into_mass2(mass_group, mass2_channels, columns)


def mass2_to_mass(
    mass2_channels: Channels,
    columns: Sequence[str],
) -> mass.off.ChannelGroup:
    """
    Create a mass ChannelGroup from a Channels object and inject analysis results.

    Parameters
    ----------
    mass2_channels : Channels
        Channels with analysis already applied.
    columns : Sequence[str]
        Names of DataFrame columns to inject into mass.

    Returns
    -------
    mass.off.ChannelGroup
        mass ChannelGroup loaded from the corresponding OFF files,
        with mass2 results injected as zero-argument recipes.

    Raises
    ------
    ImportError
        If mass is not installed.
    ValueError
        If a channel has no ``.ljh`` entry among its data sources, or if a
        requested column is absent from a channel's DataFrame.
    FileNotFoundError
        If an expected ``.off`` file does not exist next to its LJH source.
    """
    try:
        import mass.off  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "mass must be installed to use mass2_to_mass(). "
            "Install it with: pip install mass2[mass]"
        ) from exc

    off_paths: list[str] = []
    for ch in mass2_channels.channels.values():
        sources = _leaf_data_sources(ch.header)
        ljh_path = next((s for s in sources if s is not None and s.endswith(".ljh")), None)
        if ljh_path is None:
            raise ValueError(f"channel {ch.header.ch_num} has no .ljh entry in its data sources: {sources}")
        off_path = Path(ljh_path).with_suffix(".off")
        if not off_path.exists():
            raise FileNotFoundError(
                f"OFF file not found: {off_path}\n"
                "Run mass on the LJH files first to generate the OFF files."
            )
        off_paths.append(str(off_path))

    mass_group = mass.off.ChannelGroup(off_paths)
    inject_mass2_into_mass(mass2_channels, mass_group, columns)
    return mass_group
