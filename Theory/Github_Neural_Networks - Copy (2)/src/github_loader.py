from pathlib import Path

import git

from config import (
    REPOSITORY_URL,
    REPOSITORY_DIR,
    MAX_COMMITS,
)


def clone_repository():
    """
    Clone a GitHub repository using a shallow, blobless clone.

    Only the most recent MAX_COMMITS commits are initially
    downloaded. File contents from the Git history are not
    downloaded unless they are needed later.
    """

    repository_path = Path(REPOSITORY_DIR)

    # --------------------------------------------------------
    # Check whether the repository already exists
    # --------------------------------------------------------

    if repository_path.exists():

        if (repository_path / ".git").exists():

            print("Repository already exists.")
            print(f"Location: {repository_path.resolve()}")

            return

        else:
            print("Repository directory exists but is not a Git repository.")
            print("Please remove it before running the program again.")

            return

    # --------------------------------------------------------
    # Create the required directories
    # --------------------------------------------------------

    repository_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Display configuration
    # --------------------------------------------------------

    print("=" * 60)
    print("GitHub Repository Downloader")
    print("=" * 60)

    print(f"Repository : {REPOSITORY_URL}")
    print(f"Commits    : {MAX_COMMITS}")
    print(f"Destination: {repository_path.resolve()}")

    print("\nStarting clone...")
    print("This may take some time.\n")

    # --------------------------------------------------------
    # Clone repository
    # --------------------------------------------------------

    git.Repo.clone_from(
        REPOSITORY_URL,
        repository_path,
        depth=MAX_COMMITS,
        filter="blob:none",
        no_checkout=True,
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Repository cloned successfully!")
    print("=" * 60)

    print(f"Location: {repository_path.resolve()}")


if __name__ == "__main__":
    clone_repository()
