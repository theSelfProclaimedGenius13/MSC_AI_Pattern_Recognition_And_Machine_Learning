from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

# Project root:
# Github_Neural_Networks/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

REPOSITORY_DIR = RAW_DATA_DIR / "repository"


# ============================================================
# GITHUB REPOSITORY
# ============================================================

# Change ONLY this URL when using another GitHub repository.
REPOSITORY_URL = "https://github.com/blender/blender.git"


# ============================================================
# DATA COLLECTION
# ============================================================

# Number of recent commits we initially want.
MAX_COMMITS = 5000
