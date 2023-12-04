# Asyncio Tokenized Lock

[![codecov](https://codecov.io/gh/lucascicco/asyncio-tokenized-lock/graph/badge.svg?token=8F2RYP2L2L)](https://codecov.io/gh/lucascicco/asyncio-tokenized-lock)
[![PyPI](https://img.shields.io/pypi/v/asyncio-tokenized-lock)](https://pypi.org/project/asyncio-tokenized-lock/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/asyncio-tokenized-lock)

Asyncio Tokenized Lock is a Python library that provides a token-based locking mechanism for managing asynchronous locks in an asyncio environment. It introduces two main classes, `LockManager` and `TokenizedLock`, to facilitate the coordination of asynchronous tasks with associated tokens.

## Features

- **Tokenized Locking:** Locks are associated with tokens, allowing for fine-grained control over asynchronous operations.
- **Asynchronous Support:** Built on asyncio, enabling efficient coordination of asynchronous tasks.
- **Context Manager Interface:** `TokenizedLock` can be used as an asynchronous context manager for clean and concise lock management.
- **Timeout Support:** The library supports timeouts for lock acquisition.

## Installation

```shell
pip install asyncio-tokenized-lock
```

## Usage (Examples)

### Basic Usage

```python
from asyncio_tokenized_lock.lock import LockManager, TokenizedLock

manager = LockManager[str]()
lock = manager.register("example_token")

# Acquire and release the lock
async with lock:
    # Perform operations while holding the lock
    pass
```

### Timeout

#### Context Manager

```python
from asyncio_tokenized_lock.lock import LockManager, TokenizedLock

manager = LockManager[str]()
token = "example_token"
lock = manager.register(token)
lock.ctx_timeout = 1.0  # Set the timeout to 1 second
try:
    # Acquire the lock with a timeout using context manager
    async with lock:
        # Perform operations while holding the lock
        pass
except asyncio.TimeoutError:
    ...
```

#### Inline

```python
try:
    await lock.acquire(timeout=1.0)
    # Perform operations while holding the lock
except asyncio.TimeoutError:
    print("Lock acquisition timed out")
```

### Queue Concurrency

```python
import asyncio
import uuid
import logging
from asyncio_tokenized_lock.lock import LockManager, TokenizedLock

log = logging.getLogger("my-module")

async def consume_queue_safely(concurrency: int = 5, queue_size: int = 100):
    manager = LockManager[str]()
    queue = asyncio.Queue()
    put_tasks = [
        asyncio.ensure_future(queue.put(item=uuid.uuid4()))
        for _ in range(queue_size)
    ]

    async def safe_consume(queue: asyncio.Queue):
        while not queue.empty():
            item = await queue.get()
            lock = manager.register(item)

            if lock.locked:
                continue

            async with lock:
                # Perform operations with the locked item
                yield item

    # Worker class to consume from the queue safely
    @dataclass
    class Worker:
        id: str
        queue: asyncio.Queue

        async def consume(self):
            while True:
                async for item in safe_consume(self.queue):
                    # Perform operations with the item
                    log.info(f"[WORKER-{self.id}] Item {item} processed")
                break

    workers = [Worker(id=str(i), queue=queue) for i in range(concurrency)]
    consume_tasks = [asyncio.ensure_future(w.consume()) for w in workers]

    # Wait for tasks to complete
    await asyncio.wait(put_tasks)
    await asyncio.wait(consume_tasks)
```

## Testing

```shell
poetry run pytest
```

Run the provided test suite to ensure the correct behavior of the `LockManager` and `TokenizedLock` classes in different scenarios.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.

## License

This library is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
