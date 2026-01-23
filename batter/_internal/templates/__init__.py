from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]

# Example: batter/_internal/templates/build_files_orig/
BUILD_FILES_DIR = PKG_ROOT / "_internal" / "templates" / "build_files_orig"
AMBER_FILES_DIR = PKG_ROOT / "_internal" / "templates" / "amber_files_orig"
RUN_FILES_DIR = PKG_ROOT / "_internal" / "templates" / "run_files_orig"