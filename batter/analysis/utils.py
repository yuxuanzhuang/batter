"""Small numerical helpers used across :mod:`batter.analysis`."""

from __future__ import annotations

from typing import Generator, Iterable, List

import numpy as np
import pandas as pd
from loguru import logger

__all__ = [
    "SizedChunks",
    "MakeChunksWithSize",
    "MakeGroupedChunks",
    "exclude_outliers",
]


def SizedChunks(lst: Iterable[int], n: int) -> Generator[List[int], None, None]:
    """
    Yield successive ``n``-sized chunks from an iterable.

    Parameters
    ----------
    lst : Iterable[int]
        Source iterable that should be partitioned. The iterable is consumed,
        so pass a sequence (e.g. ``range``) if it needs to be reused.
    n : int
        Requested chunk size.

    Yields
    ------
    list[int]
        Consecutive slices of length ``n`` (the final chunk may be shorter).
    """
    seq = list(lst)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def MakeChunksWithSize(istart: int, istop: int, size: int) -> List[List[int]]:
    """
    Build index chunks covering ``[istart, istop)`` with approximately ``size`` elements.

    Parameters
    ----------
    istart : int
        Starting index (inclusive).
    istop : int
        Stopping index (exclusive).
    size : int
        Target chunk size prior to merging trailing fragments.

    Returns
    -------
    list[list[int]]
        Collection of contiguous index lists.
    """
    chunks = [list(chunk) for chunk in SizedChunks(range(istart, istop), size)]
    if len(chunks) > 1 and len(chunks[-1]) < (size * 2 / 3):
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def MakeGroupedChunks(ene: np.ndarray, size: int) -> List[List[int]]:
    """
    Merge adjacent chunks when their means are statistically indistinguishable.

    Parameters
    ----------
    ene : numpy.ndarray
        One-dimensional array containing the energy trace used for grouping.
    size : int
        Requested minimum chunk size prior to the adaptive merge step.

    Returns
    -------
    list[list[int]]
        List of index groups representing contiguous frames with similar means.
    """
    from scipy.stats import ttest_ind

    cidxs = MakeChunksWithSize(0, ene.shape[0], size)
    ichk = 0
    while ichk < len(cidxs) - 1:
        t_stat, p_val = ttest_ind(ene[cidxs[ichk]], ene[cidxs[ichk + 1]], equal_var=False)
        if p_val > 0.05:
            cidxs[ichk].extend(cidxs[ichk + 1])
            del cidxs[ichk + 1]
        else:
            ichk += 1
    return cidxs


def exclude_outliers(df: pd.DataFrame, iclam: int) -> pd.DataFrame:
    """
    Remove energy spikes that would destabilise MBAR fits.

    Parameters
    ----------
    df : pandas.DataFrame
        Reduced potential values with time points along the rows and lambda
        states in the columns.
    iclam : int
        Index of the reference lambda column. The algorithm analyses this
        column to decide which trajectory chunks should be discarded.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe with the same columns as ``df`` but potentially
        fewer rows if outliers were detected.

    Notes
    -----
    The implementation mirrors the heuristics used in the original
    ``fe-toolkit`` scripts: frames are chunked into ~200-sample blocks,
    grouped via a Welch t-test, and discarded whenever any lambda exhibits
    a value more than ``3σ + 1000`` kcal/mol below the reference median
    (after correcting for mixed precision offsets).
    """
    clean_df = df.replace([np.inf, -np.inf], 1e9)
    efeps = clean_df.values
    skips = [False] * efeps.shape[0]
    lams = clean_df.columns.values

    meds = np.median(efeps, axis=0) - np.median(efeps[:, iclam])

    chunk_indices = MakeGroupedChunks(efeps[:, iclam], 200)
    for idxs in chunk_indices:
        ref = np.sort(efeps[idxs, iclam])
        n = len(idxs)
        nmin = int(0.05 * n)
        nmax = int(0.95 * n)
        ref = ref[nmin:nmax]
        refm = float(np.median(ref))
        refs = float(np.std(ref))

        for i in idxs:
            for ilam, _ in enumerate(lams):
                if meds[ilam] > 10000 and efeps[i, ilam] < (refm - 3 * refs - 1000):
                    skips[i] = True
                    break
            if skips[i]:
                break

    skipped = int(np.sum(skips))
    logger.debug("exclude_outliers: removed {} frames for lambda {}", skipped, lams[iclam])
    if skipped == 0:
        return clean_df
    return clean_df[~np.array(skips)]
