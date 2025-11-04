from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

__all__ = ["Artifact", "ArtifactManifest", "ArtifactStore"]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class Artifact:
    """
    A single artifact tracked by the manifest.

    Parameters
    ----------
    name : str
        Logical name (e.g., "fe/index" or "traj/lig1.zarr").
    relpath : pathlib.Path
        Path relative to the store root.
    kind : {"file","dir"}
        File or directory artifact.
    sha256 : str
        SHA-256 of the file (empty for directories).
    size : int
        Size in bytes (files only; 0 for directories).
    meta : dict
        Free-form metadata (component, lambda, etc.).
    """
    name: str
    relpath: Path
    kind: Literal["file", "dir"] = "file"
    sha256: str = ""
    size: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


class ArtifactManifest:
    """
    In-memory manifest for a portable artifact store.

    Notes
    -----
    - Paths are **relative** to enable rebasing the store to a new root.
    - Serialize with :meth:`to_dict` / :meth:`from_dict`.
    """

    def __init__(self) -> None:
        self._items: Dict[str, Artifact] = {}

    # -------------- mutation --------------

    def add(self, art: Artifact, overwrite: bool = False) -> None:
        if art.name in self._items and not overwrite:
            raise KeyError(f"Artifact name already exists: {art.name!r}")
        self._items[art.name] = art

    # -------------- queries ---------------

    def get(self, name: str) -> Artifact:
        try:
            return self._items[name]
        except KeyError as e:
            raise KeyError(f"No artifact named {name!r} in manifest") from e

    def names(self) -> List[str]:
        return sorted(self._items.keys())

    def exists(self, name: str) -> bool:
        return name in self._items

    # -------------- i/o -------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "artifacts": [
                {
                    "name": a.name,
                    "relpath": str(a.relpath),
                    "kind": a.kind,
                    "sha256": a.sha256,
                    "size": a.size,
                    "meta": a.meta or {},
                }
                for a in self._items.values()
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactManifest":
        m = cls()
        for row in d.get("artifacts", []):
            m.add(
                Artifact(
                    name=row["name"],
                    relpath=Path(row["relpath"]),
                    kind=row.get("kind", "file"),
                    sha256=row.get("sha256", ""),
                    size=int(row.get("size", 0)),
                    meta=row.get("meta", {}) or {},
                ),
                overwrite=True,
            )
        return m


class ArtifactStore:
    """
    Portable store with a relocatable root and JSON manifest.

    Parameters
    ----------
    root : path-like
        Store root directory (e.g., a run's work directory).
    manifest_name : str
        File name for the manifest JSON under ``root`` (default: "manifest.json").

    Examples
    --------
    >>> store = ArtifactStore("work/at1r_aai")
    >>> p = store.put_file(Path("results.txt"), name="fe/latest", dst_rel=Path("fe/results.txt"))
    >>> store.save_manifest()
    >>> # move directory to a new cluster...
    >>> store2 = ArtifactStore("new_root/at1r_aai"); store2.load_manifest()
    >>> store2.path("fe/latest")
    new_root/at1r_aai/fe/results.txt
    """

    def __init__(self, root: Path | str, manifest_name: str = "manifest.json") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = ArtifactManifest()
        self.manifest_name = manifest_name

    # ---------------- core ops ----------------

    def put_file(
        self,
        src: Path,
        name: str,
        dst_rel: Optional[Path] = None,
        overwrite_manifest_entry: bool = False,
    ) -> Path:
        """
        Copy a file under the store and record it in the manifest.

        Parameters
        ----------
        src : path-like
            Source file path (must exist and be a file).
        name : str
            Logical artifact name to register under.
        dst_rel : path-like, optional
            Relative destination path. Defaults to ``name.replace('/', '_')``.
        overwrite_manifest_entry : bool
            If True, allows replacing an existing manifest entry with the same name.

        Returns
        -------
        pathlib.Path
            Absolute destination path.
        """
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"Source file does not exist or is not a file: {src}")

        dst_rel = dst_rel or Path(name.replace("/", "_"))
        if dst_rel.is_absolute():
            raise ValueError(f"dst_rel must be relative, got absolute: {dst_rel}")

        dst_abs = self.root / dst_rel
        dst_abs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_abs)  # follow symlinks by default

        sha = _sha256(dst_abs)
        size = dst_abs.stat().st_size
        self._manifest.add(
            Artifact(name=name, relpath=dst_rel, kind="file", sha256=sha, size=size),
            overwrite=overwrite_manifest_entry,
        )
        return dst_abs

    def put_dir(
        self,
        src_dir: Path,
        name: str,
        dst_rel: Optional[Path] = None,
        overwrite_manifest_entry: bool = False,
    ) -> Path:
        """
        Copy a directory under the store and record it in the manifest.

        Notes
        -----
        - No per-file hashing; use :meth:`put_file` for critical files.
        """
        src_dir = Path(src_dir)
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Source directory does not exist or is not a directory: {src_dir}")

        dst_rel = dst_rel or Path(name)
        if dst_rel.is_absolute():
            raise ValueError(f"dst_rel must be relative, got absolute: {dst_rel}")

        dst_abs = self.root / dst_rel
        if dst_abs.exists():
            shutil.rmtree(dst_abs)
        shutil.copytree(src_dir, dst_abs)
        self._manifest.add(Artifact(name=name, relpath=dst_rel, kind="dir"), overwrite=overwrite_manifest_entry)
        return dst_abs

    def path(self, name: str) -> Path:
        """Resolve an artifact name to an **absolute** path under the current root."""
        return self.root / self._manifest.get(name).relpath

    # ---------------- manifest i/o ----------------

    def save_manifest(self) -> Path:
        """Write the manifest JSON under ``root`` (atomic)."""
        p = self.root / self.manifest_name
        tmp = p.with_suffix(p.suffix + ".tmp")
        data = json.dumps(self._manifest.to_dict(), indent=2)
        tmp.write_text(data)
        os.replace(tmp, p)  # atomic on POSIX
        return p

    def load_manifest(self) -> None:
        """Load the manifest JSON from ``root``."""
        p = self.root / self.manifest_name
        if not p.is_file():
            raise FileNotFoundError(f"Manifest not found at {p}")
        self._manifest = ArtifactManifest.from_dict(json.loads(p.read_text()))

    # ---------------- portability ----------------

    def rebase(self, new_root: Path | str) -> "ArtifactStore":
        """
        Create a new store view with the same manifest but a different root.

        Parameters
        ----------
        new_root : path-like
            Target root directory.

        Returns
        -------
        ArtifactStore
            New store pointing to ``new_root``.
        """
        s = ArtifactStore(new_root, self.manifest_name)
        s._manifest = self._manifest  # share view
        return s