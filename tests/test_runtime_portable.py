from pathlib import Path

from batter.runtime.portable import ArtifactStore


def test_list_artifacts_filters(tmp_path):
    store_root = tmp_path / "store"
    store_root.mkdir()
    store = ArtifactStore(store_root)

    src1 = tmp_path / "src1.txt"
    src1.write_text("hello")
    src2 = tmp_path / "src2.txt"
    src2.write_text("world")

    store.put_file(src1, name="logs/latest")
    store.put_file(src2, name="fe/results")

    all_artifacts = store.list_artifacts()
    assert {a.name for a in all_artifacts} == {"logs/latest", "fe/results"}

    logs_only = store.list_artifacts(prefix="logs")
    assert [a.name for a in logs_only] == ["logs/latest"]

    files_only = store.list_artifacts(kind="file")
    assert len(files_only) == 2
