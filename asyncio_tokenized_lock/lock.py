import asyncio
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar
from weakref import WeakValueDictionary

__all__ = ["LockManager", "TokenizedLock"]

T = TypeVar("T", str, int, tuple)


@dataclass
class LockManager(Generic[T]):
    """Manages TokenizedLock instances using tokens."""

    _tokens: WeakValueDictionary[T, "TokenizedLock[T]"] = field(
        default_factory=WeakValueDictionary
    )

    def __repr__(self) -> str:
        return f"LockManager({list(self._tokens.keys())})"

    def __len__(self) -> int:
        return len(self._tokens)

    def _get_lock(self, token: T) -> Optional["TokenizedLock[T]"]:
        return self._tokens.get(token)

    def register(self, token: T) -> "TokenizedLock[T]":
        """Register a TokenizedLock instance associated with a token.

        If a TokenizedLock instance with the given token already exists, it will be
        returned. Otherwise, a new TokenizedLock instance will be created and
        registered.

        Parameters
        ----------
        - token: The token associated with the TokenizedLock instance.

        Returns
        -------
        - TokenizedLock: The registered or existing TokenizedLock instance.
        """
        lock = self._get_lock(token)
        if not lock:
            lock = TokenizedLock[T](token=token, manager=self)
            self._tokens[token] = lock
        return lock

    def release_all(self, *, safe: bool = True) -> None:
        """Releases all TokenizedLock instances managed by the LockManager."""
        for lock in list(self._tokens.values()):
            lock.release(safe=safe)
            self._unregister(lock.token)

    def _unregister(self, token: T) -> None:
        """Unregisters a TokenizedLock instance associated with a token if it's released."""
        lock = self._get_lock(token)
        if lock and lock._released:
            self._tokens.pop(token)


@dataclass
class TokenizedLock(Generic[T]):
    """Represents a lock associated with a token."""

    token: T
    manager: LockManager
    _ctx_timeout: Optional[float] = field(default=None, init=False)
    _released: bool = field(default=False, init=False)
    _iolock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        init=False,
    )

    def __repr__(self) -> str:
        extra = "locked" if self.locked else "unlocked"
        waiters = getattr(self._iolock, "_waiters", [])
        if waiters:
            extra = f"{extra}, waiters:{len(waiters)}"
        return f"<TokenizedLock {self.token!r} {extra}>"

    @property
    def ctx_timeout(self) -> Optional[float]:
        return self._ctx_timeout

    @ctx_timeout.setter
    def ctx_timeout(self, value: Optional[float]) -> None:
        self._ctx_timeout = value

    @property
    def locked(self) -> bool:
        """Checks if the TokenizedLock instance is currently locked."""
        return self._iolock.locked()

    def _release(self) -> None:
        """Releases the TokenizedLock instance."""
        self._iolock.release()
        waiters = getattr(self._iolock, "_waiters", [])
        if waiters is None or len(waiters) < 1:
            self._released = True
            self.manager._unregister(self.token)

    def release(self, *, safe: bool = False) -> None:
        """Releases the TokenizedLock instance.

        Parameters
        ----------
        - safe (bool, optional): If True, the release operation is conditional.
          It will only release the lock if it is still locked and has not been
          manually released before. If False, the lock is unconditionally released.

        Notes
        -----
        - When using `safe=True`, the method checks whether the lock is still locked
          and has not been manually released to avoid unintended releases.

        Example:
        >>> lock = TokenizedLock(token="example", manager=manager)
        >>> await lock.acquire()
        >>> # ... do work ...
        >>> lock.release(safe=True)  # Releases the lock only if it's still held

        """
        if safe:
            if not self._released and self.locked:
                self._release()
            return
        self._release()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquires the TokenizedLock instance asynchronously.

        Parameters
        ----------
        - timeout (float | None): The maximum time, in seconds, to wait for the lock.
          If None, it will wait indefinitely.

        Returns
        -------
        - bool: True if the lock is acquired successfully within the specified timeout,
          False otherwise.

        Raises
        ------
        - TimeoutError: If the lock cannot be acquired within the specified timeout.
        """
        lock = self.manager.register(self.token)
        try:
            acquired = await asyncio.wait_for(lock._iolock.acquire(), timeout=timeout)
        except TimeoutError as e:
            raise TimeoutError(
                f"Failed to acquire lock within the specified timeout ({timeout} seconds)."
            ) from e
        else:
            return acquired

    async def __aenter__(self) -> None:
        """Asynchronous context manager entry."""
        await self.acquire(timeout=self.ctx_timeout)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Asynchronous context manager exit."""
        self.release()
