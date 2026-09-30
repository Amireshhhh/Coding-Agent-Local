"""Git checkpoints (Phase 3): snapshot the working tree before edits; ``/undo`` restores it.

Snapshots never touch the user's index, HEAD or branches: the tree is written through a temporary
index (``GIT_INDEX_FILE``), wrapped in a commit object and kept under ``refs/agent/checkpoints/<n>``.
Untracked files are included (``.gitignore`` is respected). Restoring rewrites tracked+untracked
files to the snapshot and deletes files created after it (ignored files are never touched).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

REF_PREFIX = "refs/agent/checkpoints/"


class GitError(RuntimeError):
    """A git command failed."""


class Checkpoints:
    """Create and restore working-tree snapshots of a git repository."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def _git(self, *args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
        proc = await asyncio.create_subprocess_exec(
            "git", *args, cwd=str(self.root), env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {err.decode(errors='replace').strip()[:300]}")
        return out.decode(errors="replace").strip()

    async def available(self) -> bool:
        """True if ``root`` is inside a git work tree."""
        try:
            return await self._git("rev-parse", "--is-inside-work-tree") == "true"
        except (GitError, FileNotFoundError):
            return False

    async def _tree_of_worktree(self) -> str:
        fd, idx = tempfile.mkstemp(prefix="agent-index-")
        os.close(fd)
        os.unlink(idx)
        env = {"GIT_INDEX_FILE": idx}
        try:
            await self._git("add", "-A", ".", env=env)
            return await self._git("write-tree", env=env)
        finally:
            if os.path.exists(idx):
                os.unlink(idx)

    async def create(self, label: str = "checkpoint") -> str:
        """Snapshot the working tree; returns the checkpoint ref name."""
        tree = await self._tree_of_worktree()
        head = await self._git("rev-parse", "--verify", "-q", "HEAD", check=False)
        args = ["commit-tree", tree, "-m", f"agent {label}"]
        if head:
            args[2:2] = ["-p", head]
        env = {"GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "agent@localhost",
               "GIT_COMMITTER_NAME": "agent", "GIT_COMMITTER_EMAIL": "agent@localhost"}
        commit = await self._git(*args, env=env)
        n = len(await self.list()) + 1
        ref = f"{REF_PREFIX}{n:04d}"
        await self._git("update-ref", ref, commit)
        return ref

    async def list(self) -> list[str]:
        """Checkpoint refs, oldest first."""
        out = await self._git("for-each-ref", "--format=%(refname)", REF_PREFIX)
        return sorted(line for line in out.splitlines() if line)

    async def _files(self, tree: str) -> set[str]:
        out = await self._git("ls-tree", "-r", "--name-only", "-z", tree)
        return {p for p in out.split("\0") if p}

    async def restore(self, ref: str | None = None) -> tuple[str, int, int]:
        """Restore the latest (or given) checkpoint and delete it.

        Returns ``(ref, files_written, files_deleted)``.

        Raises:
            GitError: no checkpoint exists or git failed.
        """
        refs = await self.list()
        if not refs:
            raise GitError("no checkpoints to restore")
        ref = ref or refs[-1]
        target_tree = await self._git("rev-parse", f"{ref}^{{tree}}")
        now_tree = await self._tree_of_worktree()
        target_files = await self._files(target_tree)
        now_files = await self._files(now_tree)
        fd, idx = tempfile.mkstemp(prefix="agent-index-")
        os.close(fd)
        os.unlink(idx)
        env = {"GIT_INDEX_FILE": idx}
        try:
            await self._git("read-tree", target_tree, env=env)
            await self._git("checkout-index", "-a", "-f", env=env)
        finally:
            if os.path.exists(idx):
                os.unlink(idx)
        deleted = 0
        for rel in sorted(now_files - target_files):
            p = self.root / rel
            if p.is_file() or p.is_symlink():
                p.unlink()
                deleted += 1
        await self._git("update-ref", "-d", ref)
        return ref, len(target_files), deleted
