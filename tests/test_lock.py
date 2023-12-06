import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass

import pytest

from asyncio_tokenized_lock.lock import LockManager, TokenizedLock


async def consume_queue_safely(
    concurrency: int = 5,
    queue_size: int = 100,
):
    manager = LockManager[str]()
    queue = asyncio.Queue()
    acquire_counter: dict[str, int] = defaultdict(lambda: 0)

    put_tasks = [
        asyncio.ensure_future(queue.put(item=uuid.uuid4())) for _ in range(queue_size)
    ]

    async def safe_consume(queue: asyncio.Queue):
        while not queue.empty():
            item = await queue.get()
            lock = manager.register(item)

            if lock.locked:
                continue

            async with lock:
                acquire_counter[item] += 1
                yield item

    @dataclass
    class Worker:
        id: str
        queue: asyncio.Queue

        async def consume(self):
            while True:
                async for _ in safe_consume(self.queue):
                    await asyncio.sleep(0.1)
                break

    workers = [Worker(id=str(i), queue=queue) for i in range(concurrency)]
    consume_tasks = [asyncio.ensure_future(w.consume()) for w in workers]
    await asyncio.wait(put_tasks)
    await asyncio.wait(consume_tasks)

    assert sum(acquire_counter.values()) == queue_size
    assert len(manager._locks_by_token) == 0


@pytest.mark.asyncio()
async def test_lock_manager_basic_operations():
    manager = LockManager[str]()
    lock = manager.register("first")
    assert not lock.locked
    await lock.acquire()
    assert lock.locked
    lock.release()
    assert not lock.locked


@pytest.mark.asyncio()
async def test_lock_manager_context_manager():
    manager = LockManager[str]()
    lock = manager.register("test_context")
    async with lock:
        assert lock.locked
    assert not lock.locked


@pytest.mark.asyncio()
async def test_lock_manager_with_tokens():
    manager = LockManager[int]()
    lock_1 = manager.register(123)
    lock_2 = manager.register(456)
    assert not lock_1.locked
    assert not lock_2.locked
    await lock_1.acquire()
    assert lock_1.locked
    assert not lock_2.locked
    lock_1.release()
    assert not lock_1.locked


@pytest.mark.asyncio()
async def test_lock_manager_timeout():
    manager = LockManager[tuple]()
    lock = manager.register((1, 2, 3))
    assert not lock.locked
    await lock.acquire(timeout=0.1)
    assert lock.locked
    with pytest.raises(asyncio.TimeoutError):
        await lock.acquire(timeout=0.1)


@pytest.mark.asyncio()
async def test_lock_manager_context_manager_timeout_error():
    loop = asyncio.get_event_loop()

    manager = LockManager[tuple]()
    lock = manager.register((1, 2, 3))
    lock.ctx_timeout = 0.5
    assert not lock.locked
    await lock.acquire()
    assert lock.locked
    loop.call_later(delay=1.5, callback=lock.release)
    with pytest.raises(asyncio.TimeoutError):
        async with lock:
            pass


@pytest.mark.asyncio()
async def test_lock_manager_context_manager_timeout_release():
    loop = asyncio.get_event_loop()

    manager = LockManager[tuple]()
    lock = manager.register((1, 2, 3))
    lock.ctx_timeout = 1
    assert not lock.locked
    await lock.acquire()
    assert lock.locked
    loop.call_later(delay=0.5, callback=lock.release)
    async with lock:
        pass


def test_lock_manager_concurrency():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(consume_queue_safely())


@pytest.mark.asyncio()
async def test_lock_manager_len():
    manager = LockManager[int]()
    f_lock = manager.register(1)
    s_lock = manager.register(2)
    assert len(manager) == 2
    assert len(manager._locks_by_token) == 2
    await f_lock.acquire()
    await s_lock.acquire()
    manager.release_all()
    assert len(manager) == 0
    assert len(manager._locks_by_token) == 0


@pytest.mark.asyncio()
async def test_lock_manager_repr():
    manager = LockManager[str]()
    f_lock = manager.register("first")
    s_lock = manager.register("second")
    assert repr(manager) == f"LockManager({[f_lock.token, s_lock.token]})"


@pytest.mark.asyncio()
async def test_tokenized_lock_repr():
    manager = LockManager[str]()
    f_lock = manager.register("first")

    num_acquirers = 10

    async def acquirer(lock: TokenizedLock):
        await lock.acquire()
        await asyncio.sleep(0.1)

    acquirers = [asyncio.ensure_future(acquirer(f_lock)) for _ in range(num_acquirers)]
    await asyncio.wait(acquirers, return_when=asyncio.FIRST_COMPLETED)

    def get_lock_repr(token: str, locked: bool, waiters: int) -> str:
        extra = "locked" if locked else "unlocked"
        if waiters > 0:
            extra = f"{extra}, waiters:{waiters}"
        return f"<TokenizedLock {token!r} {extra}>"

    waiters = num_acquirers - 1
    f_lock_repr = get_lock_repr(token="first", locked=True, waiters=waiters)
    assert repr(f_lock) == f_lock_repr


@pytest.mark.asyncio()
async def test_lock_manager_weak_value_dict_ref():
    manager = LockManager[int]()
    lock = manager.register(1)
    await lock.acquire()
    assert len(manager) == 1
    await asyncio.sleep(1)
    del lock
    assert len(manager) == 0
    f_lock = manager.register(1)
    assert not f_lock.locked
    await f_lock.acquire()
    await asyncio.sleep(1)
    assert len(manager) == 1
