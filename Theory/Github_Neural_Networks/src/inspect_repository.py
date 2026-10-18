from pathlib import Path

import git

from config import REPOSITORY_DIR, MAX_COMMITS


def inspect_repository():
    repository_path = Path(REPOSITORY_DIR)

    # --------------------------------------------------------
    # Check repository exists
    # --------------------------------------------------------

    if not repository_path.exists():
        print("Repository does not exist.")
        print(f"Expected location: {repository_path.resolve()}")
        return

    if not (repository_path / ".git").exists():
        print("The directory exists, but it is not a Git repository.")
        return

    # --------------------------------------------------------
    # Open repository
    # --------------------------------------------------------

    repo = git.Repo(repository_path)

    print("=" * 70)
    print("BLENDER GIT REPOSITORY INSPECTION")
    print("=" * 70)

    print(f"Repository location : {repository_path.resolve()}")
    print(f"Current commit      : {repo.head.commit.hexsha}")

    # --------------------------------------------------------
    # Count commits
    # --------------------------------------------------------

    print("\nCounting commits...")

    commits = list(repo.iter_commits())

    print(f"Commits available   : {len(commits)}")

    # --------------------------------------------------------
    # Show newest and oldest commit
    # --------------------------------------------------------

    if commits:

        newest = commits[0]
        oldest = commits[-1]

        print("\n" + "-" * 70)
        print("NEWEST COMMIT")
        print("-" * 70)

        print(f"SHA       : {newest.hexsha}")
        print(f"Author    : {newest.author.name}")
        print(f"Date      : {newest.committed_datetime}")
        print(f"Message   : {newest.message.strip()}")

        print("\n" + "-" * 70)
        print("OLDEST COMMIT IN DOWNLOADED HISTORY")
        print("-" * 70)

        print(f"SHA       : {oldest.hexsha}")
        print(f"Author    : {oldest.author.name}")
        print(f"Date      : {oldest.committed_datetime}")
        print(f"Message   : {oldest.message.strip()}")

    # --------------------------------------------------------
    # Show sample commits
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"FIRST 10 COMMITS")
    print("=" * 70)

    for number, commit in enumerate(commits[:10], start=1):

        message = commit.message.strip().split("\n")[0]

        print(
            f"{number:2}. "
            f"{commit.hexsha[:10]} | "
            f"{commit.committed_datetime.date()} | "
            f"{message}"
        )


if __name__ == "__main__":
    inspect_repository()
