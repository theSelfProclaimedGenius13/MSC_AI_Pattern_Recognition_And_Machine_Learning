from pathlib import Path
from argparse import ArgumentParser
from collections import Counter
import json

import git
import pandas as pd
from pathlib import PurePosixPath

from config import (
    REPOSITORY_DIR,
    RAW_DATA_DIR,
    MAX_COMMITS,
)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def get_arguments():
    parser = ArgumentParser(
        description="Extract structured Git commit data."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_COMMITS,
        help=f"Maximum number of non-merge commits "
             f"(default: {MAX_COMMITS})"
    )

    return parser.parse_args()


# ============================================================
# FILE INFORMATION
# ============================================================

def get_file_information(commit, parent):
    """
    Extract lightweight information about files changed
    between the parent and the current commit.

    IMPORTANT:
    This intentionally does NOT calculate insertions/deletions.
    That keeps the extractor fast for a blobless repository.
    """

    file_rows = []

    # Compare parent tree -> current commit tree.
    #
    # create_patch=False means we only ask Git for structural
    # information about the changed files.
    diffs = parent.diff(
        commit,
        create_patch=False
    )

    for diff in diffs:

        change_type = diff.change_type

        # Added/modified files normally have b_path.
        # Deleted files may only have a_path.
        file_path = diff.b_path or diff.a_path

        if file_path is None:
            continue

        path = PurePosixPath(file_path)

        # ----------------------------------------------------
        # File extension
        # ----------------------------------------------------

        extension = path.suffix.lower()

        if extension == "":
            extension = "[no_extension]"

        # ----------------------------------------------------
        # Top-level directory
        # ----------------------------------------------------

        parts = file_path.split("/")

        if len(parts) > 1:
            top_level_directory = parts[0]
        else:
            top_level_directory = "[root]"

        # ----------------------------------------------------
        # Directory depth
        # ----------------------------------------------------

        directory_depth = len(parts) - 1

        # ----------------------------------------------------
        # File-level row
        # ----------------------------------------------------

        file_rows.append(
            {
                "commit_sha": commit.hexsha,
                "file_path": file_path,
                "change_type": change_type,
                "extension": extension,
                "top_level_directory": top_level_directory,
                "directory_depth": directory_depth,
            }
        )

    return file_rows


# ============================================================
# COMMIT EXTRACTION
# ============================================================

def extract_commits(limit):

    repository_path = Path(REPOSITORY_DIR)

    if not repository_path.exists():
        raise FileNotFoundError(
            f"Repository not found:\n"
            f"{repository_path.resolve()}"
        )

    if not (repository_path / ".git").exists():
        raise ValueError(
            f"Directory is not a Git repository:\n"
            f"{repository_path.resolve()}"
        )

    repo = git.Repo(repository_path)

    commit_rows = []
    file_rows = []

    processed = 0
    inspected = 0
    skipped_merges = 0

    print("=" * 70)
    print("FAST GIT COMMIT DATA EXTRACTION")
    print("=" * 70)

    print(f"Repository : {repository_path.resolve()}")
    print(f"Limit      : {limit}")
    print()
    print("Using lightweight tree/diff information.")
    print("Line insertion/deletion statistics are intentionally skipped.")
    print()

    # ========================================================
    # ITERATE THROUGH COMMITS
    # ========================================================

    for commit in repo.iter_commits():

        inspected += 1

        # ----------------------------------------------------
        # Ignore merge commits
        # ----------------------------------------------------

        if len(commit.parents) != 1:
            skipped_merges += 1
            continue

        parent = commit.parents[0]

        # ----------------------------------------------------
        # Extract changed files
        # ----------------------------------------------------

        current_file_rows = get_file_information(
            commit,
            parent
        )

        # If Git reports no changed files, skip it.
        if not current_file_rows:
            continue

        file_rows.extend(current_file_rows)

        # ----------------------------------------------------
        # Change-type counts
        # ----------------------------------------------------

        change_counter = Counter(
            row["change_type"]
            for row in current_file_rows
        )

        # ----------------------------------------------------
        # Extension information
        # ----------------------------------------------------

        extensions = sorted(
            set(
                row["extension"]
                for row in current_file_rows
            )
        )

        extension_counter = Counter(
            row["extension"]
            for row in current_file_rows
        )

        # ----------------------------------------------------
        # Directory information
        # ----------------------------------------------------

        directories = sorted(
            set(
                row["top_level_directory"]
                for row in current_file_rows
            )
        )

        # ----------------------------------------------------
        # Path depth
        # ----------------------------------------------------

        depths = [
            row["directory_depth"]
            for row in current_file_rows
        ]

        average_directory_depth = (
            sum(depths) / len(depths)
        )

        # ----------------------------------------------------
        # Commit message
        # ----------------------------------------------------

        full_message = commit.message.strip()

        message_lines = full_message.splitlines()

        subject = (
            message_lines[0]
            if message_lines
            else ""
        )

        # ----------------------------------------------------
        # Commit date
        # ----------------------------------------------------

        committed_datetime = (
            commit.committed_datetime.isoformat()
        )

        # ----------------------------------------------------
        # Commit-level row
        # ----------------------------------------------------

        commit_rows.append(
            {
                "commit_sha": commit.hexsha,

                "parent_sha": parent.hexsha,

                "author_name": commit.author.name,

                "committed_datetime": committed_datetime,

                "subject": subject,

                "message": full_message,

                # ------------------------------
                # Structural features
                # ------------------------------

                "files_changed": len(current_file_rows),

                "files_added": change_counter.get("A", 0),

                "files_deleted": change_counter.get("D", 0),

                "files_modified": change_counter.get("M", 0),

                "files_renamed": change_counter.get("R", 0),

                "files_copied": change_counter.get("C", 0),

                "unique_extensions": len(extensions),

                "unique_directories": len(directories),

                "average_directory_depth":
                    average_directory_depth,

                # ------------------------------
                # JSON summaries
                # ------------------------------

                "extensions":
                    json.dumps(
                        extension_counter,
                        sort_keys=True
                    ),

                "top_level_directories":
                    json.dumps(
                        directories
                    ),

                "parent_count": len(commit.parents),
            }
        )

        processed += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed % 100 == 0:

            print(
                f"Processed {processed:,} commits "
                f"| inspected {inspected:,}"
            )

        # ----------------------------------------------------
        # Stop when limit reached
        # ----------------------------------------------------

        if processed >= limit:
            break

    # ========================================================
    # CREATE DATAFRAMES
    # ========================================================

    commits_df = pd.DataFrame(
        commit_rows
    )

    files_df = pd.DataFrame(
        file_rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    commits_file = (
        RAW_DATA_DIR / "commits.csv"
    )

    files_file = (
        RAW_DATA_DIR / "commit_files.csv"
    )

    commits_df.to_csv(
        commits_file,
        index=False
    )

    files_df.to_csv(
        files_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Commits extracted : "
        f"{len(commits_df):,}"
    )

    print(
        f"Files extracted   : "
        f"{len(files_df):,}"
    )

    print(
        f"Merge commits     : "
        f"{skipped_merges:,}"
    )

    print(
        f"Commits inspected : "
        f"{inspected:,}"
    )

    print()
    print(
        f"Commit data:"
    )

    print(
        commits_file.resolve()
    )

    print()
    print(
        f"File data:"
    )

    print(
        files_file.resolve()
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = get_arguments()

    extract_commits(
        limit=args.limit
    )
