"""
Concurrency strategies (asyncio, threading).

References: 
01_applicazioni_asincrone_con_asyncio.py 
assistenza_clienti.py
01 Introduzione alla Programmazione Asincrona.pptx 
01 introduzione alla programmazione concorrente.pptx 03 Threading in Python.pptx

Interchangeable strategies that run a list of jobs under a concurrency limit.
A *job* is a zero-argument coroutine function. Every strategy calls ``on_result(index, value)`` as soon as a job finishes, 
so callers see results as they arrive; the index lets them put each one back in its original position.
"""

import asyncio
import threading
from typing import Any, Awaitable, Callable, List

Job = Callable[[], Awaitable[Any]]  
OnResult = Callable[[int, Any], None]
# Callable[[...], ...] describes the function’s argument and return types. 
# Awaitable[Any] means the returned value can be awaited; Any means its type isn’t specified, as in Typescript.

class ConcurrencyStrategy:
    """How a batch of jobs is executed. Subclasses implement ``run``."""

    name = ""

    def __init__(self, max_concurrency: int = 10) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.max_concurrency = max_concurrency

    async def run(self, jobs: List[Job], on_result: OnResult) -> None:
        """Run all jobs, reporting each result immediately."""
        raise NotImplementedError


class SequentialStrategy(ConcurrencyStrategy):
    """One job at a time"""

    name = "sequential"

    def __init__(self, max_concurrency: int = 1) -> None:
            super().__init__(1)

    async def run(self, jobs: List[Job], on_result: OnResult) -> None:
        """
        enumerate(jobs) gives each job its position (index) and the job function.
        await job() runs that async job and waits for it to finish.
        Its result is passed to on_result(index, result).        
        """
        for index, job in enumerate(jobs):
            on_result(index, await job())


class AsyncioStrategy(ConcurrencyStrategy):
    """
    Main strategy: one coroutine per job, a semaphore caps the jobs in flight.
    Each job reports its own result the moment it finishes (inside the
    semaphore), so results are available as they arrive. 
    ``asyncio.gather`` waits for all of them; 
    if the batch is cancelled, ``gather`` cancels every unfinished job.
    """

    name = "asyncio"

    async def run(self, jobs: List[Job], on_result: OnResult) -> None:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def guarded(index: int, job: Job) -> None:
            async with semaphore:
                on_result(index, await job())

        await asyncio.gather(*(guarded(i, job) for i, job in enumerate(jobs)))


class ThreadingStrategy(ConcurrencyStrategy):
    """Alternative strategy: one thread per job, a threading semaphore caps them.

    Each thread runs its job in its own event loop (``asyncio.run``) because the
    provider calls are coroutines. ``on_result`` is called under a lock, since
    several threads finish at the same time.

    Limits compared with asyncio: ``run`` waits for the threads with ``join`` and
    so blocks the calling event loop, and a running thread cannot be interrupted.
    ``cancel()`` only prevents jobs that have not started yet.
    """

    name = "threading"

    def __init__(self, max_concurrency: int = 10) -> None:
        super().__init__(max_concurrency)
        self._cancelled = False

    def cancel(self) -> None:
        """Skip every job that has not started yet."""
        
        self._cancelled = True

    async def run(self, jobs: List[Job], on_result: OnResult) -> None:
        self._cancelled = False
        slots = threading.Semaphore(self.max_concurrency)
        report_lock = threading.Lock()
        crashes: List[BaseException] = []

        def worker(index: int, job: Job) -> None:
            with slots:
                if self._cancelled:
                    return
                try:
                    value = asyncio.run(job())
                except BaseException as error:  # reported after the join
                    crashes.append(error)
                    return
                with report_lock:
                    on_result(index, value)

        threads = [
            threading.Thread(target=worker, args=(i, job), name=f"orch-{i}")
            for i, job in enumerate(jobs)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if crashes:
            raise crashes[0]


STRATEGIES = {
    cls.name: cls for cls in (SequentialStrategy, AsyncioStrategy, ThreadingStrategy)
}


def create_strategy(name: str, max_concurrency: int) -> ConcurrencyStrategy:
    """Build a strategy by name (``sequential``, ``asyncio`` or ``threading``)."""
    if name not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {name!r}: choose one of {', '.join(sorted(STRATEGIES))}"
        )
    return STRATEGIES[name](max_concurrency)

