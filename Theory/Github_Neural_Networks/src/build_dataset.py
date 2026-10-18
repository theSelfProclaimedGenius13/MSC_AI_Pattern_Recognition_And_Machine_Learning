# src/build_dataset.py

import json
from pathlib import Path

import pandas as pd

from config import DATA_DIR


# ============================================================
# Configuration
# ============================================================

FUTURE_WINDOW_DAYS = 30

COMMITS_FILE = DATA_DIR / "raw" / "commits.csv"
COMMIT_FILES_FILE = DATA_DIR / "raw" / "commit_files.csv"

OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "commit_dataset.csv"


# ============================================================
# Bug-fix detector
# ============================================================

BUG_FIX_PREFIXES = (
    "fix:",
    "fix ",
    "fix(",
    "fix/",
    "bugfix:",
    "bug fix:",
)


def is_bug_fix(subject):
    """
    Determine whether a commit appears to be a bug-fixing commit
    based on its commit subject.
    """

    if not isinstance(subject, str):
        return False

    text = subject.strip().lower()

    if text.startswith(BUG_FIX_PREFIXES):
        return True

    if text.startswith("fixed:"):
        return True

    return False


# ============================================================
# Main dataset construction
# ============================================================

def build_dataset():

    print("=" * 60)
    print("BUILDING ML DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    commits = pd.read_csv(COMMITS_FILE)
    commit_files = pd.read_csv(COMMIT_FILES_FILE)

    print(f"Commits loaded      : {len(commits):,}")
    print(f"File records loaded : {len(commit_files):,}")

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    commits["committed_datetime"] = pd.to_datetime(
        commits["committed_datetime"],
        errors="coerce",
        utc=True
    )

    # Remove commits with invalid dates
    commits = commits.dropna(subset=["committed_datetime"]).copy()

    commits = commits.sort_values("committed_datetime").reset_index(drop=True)

    # --------------------------------------------------------
    # Detect bug-fixing commits
    # --------------------------------------------------------

    commits["is_bug_fix"] = commits["subject"].apply(is_bug_fix)

    bug_fix_commits = commits[commits["is_bug_fix"]].copy()

    print(f"Bug-fixing commits : {len(bug_fix_commits):,}")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We can only label a commit if its complete 30-day future
    # window exists inside our observed repository history.
    # --------------------------------------------------------

    max_observed_date = commits["committed_datetime"].max()

    cutoff_date = (
        max_observed_date
        - pd.Timedelta(days=FUTURE_WINDOW_DAYS)
    )

    eligible_mask = (
        commits["committed_datetime"] <= cutoff_date
    )

    eligible_commits = commits[eligible_mask].copy()
    excluded_commits = commits[~eligible_mask].copy()

    print()
    print(f"Latest observed commit : {max_observed_date}")
    print(f"30-day cutoff          : {cutoff_date}")
    print(f"Eligible commits       : {len(eligible_commits):,}")
    print(f"Excluded recent commits: {len(excluded_commits):,}")

    # --------------------------------------------------------
    # Create mapping:
    #
    # file -> dates on which that file was modified by a
    # bug-fixing commit
    # --------------------------------------------------------

    bug_fix_sha = set(
        bug_fix_commits["commit_sha"]
    )

    bug_fix_files = commit_files[
        commit_files["commit_sha"].isin(bug_fix_sha)
    ].copy()

    bug_fix_files = bug_fix_files.merge(
        commits[
            ["commit_sha", "committed_datetime"]
        ],
        on="commit_sha",
        how="left"
    )

    file_to_fix_dates = {}

    for row in bug_fix_files.itertuples(index=False):

        file_path = row.file_path
        fix_date = row.committed_datetime

        if file_path not in file_to_fix_dates:
            file_to_fix_dates[file_path] = []

        file_to_fix_dates[file_path].append(fix_date)

    # Sort dates for each file
    for file_path in file_to_fix_dates:
        file_to_fix_dates[file_path].sort()

    # --------------------------------------------------------
    # Create commit -> files mapping
    # --------------------------------------------------------

    commit_to_files = (
        commit_files
        .groupby("commit_sha")["file_path"]
        .apply(set)
        .to_dict()
    )

    # --------------------------------------------------------
    # Create target
    # --------------------------------------------------------

    labels = []

    future_window = pd.Timedelta(days=FUTURE_WINDOW_DAYS)

    for row in eligible_commits.itertuples(index=False):

        commit_sha = row.commit_sha
        commit_date = row.committed_datetime

        files = commit_to_files.get(commit_sha, set())

        window_end = commit_date + future_window

        future_bug_fix = 0

        for file_path in files:

            fix_dates = file_to_fix_dates.get(
                file_path,
                []
            )

            for fix_date in fix_dates:

                if commit_date < fix_date <= window_end:
                    future_bug_fix = 1
                    break

            if future_bug_fix:
                break

        labels.append(future_bug_fix)

    eligible_commits["future_bug_fix"] = labels

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    final_columns = [
        "commit_sha",
        "future_bug_fix",

        "files_changed",
        "files_added",
        "files_deleted",
        "files_modified",
        "files_renamed",
        "files_copied",

        "unique_extensions",
        "unique_directories",
        "average_directory_depth",

        "committed_datetime",
        "author_name",
    ]

    dataset = eligible_commits[final_columns].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    positive = int(
        dataset["future_bug_fix"].sum()
    )

    negative = len(dataset) - positive

    print()
    print("=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(f"Total labeled commits : {len(dataset):,}")
    print(f"Positive              : {positive:,}")
    print(f"Negative              : {negative:,}")

    print(
        f"Positive percentage   : "
        f"{positive / len(dataset) * 100:.2f}%"
    )

    print()
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()
