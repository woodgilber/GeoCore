import os
from typing import List, Optional

from git import Diff, Repo


def check_for_commits() -> Optional[List[Diff]]:
    """Check for uncommited files

    Returns:
        Optional[List[Diff]]: either None if everything was commited or a list of diffs if not
    """
    repo = Repo.init(os.getcwd())
    # check for changes not staged for commit
    diffs = repo.index.diff(None)
    if diffs:
        return diffs

    # check for changes to be commited
    diffs = repo.index.diff(repo.head.commit)
    if diffs:
        return diffs
    return
