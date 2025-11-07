import numpy as np

from batter.analysis.remd import RemdLog


def test_remd_analysis_reports_round_trips():
    trajectory = np.array(
        [
            [1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2],
        ]
    )
    stats = RemdLog._remd_analysis(trajectory, ARs=[0.5])

    assert stats["Average single pass steps:"] == 1.0
    assert stats["Round trips per replica:"] == 2.0
    assert stats["Total round trips:"] == 4.0
    assert stats["neighbor_acceptance_ratio"] == [0.5]


def test_remd_analysis_handles_missing_passes():
    trajectory = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]
    )
    stats = RemdLog._remd_analysis(trajectory, ARs=[])

    assert stats["Average single pass steps:"] == 1.0e8
    assert stats["Round trips per replica:"] == 0.0
    assert stats["Total round trips:"] == 0.0
    assert stats["neighbor_acceptance_ratio"] == []
