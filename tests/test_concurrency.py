"""Tests for the concurrency strategies."""

import asyncio
import threading
import unittest

from calls_coordinator.concurrency import (
    AsyncioStrategy,
    SequentialStrategy,
    ThreadingStrategy,
    create_strategy,
)
from tests.helpers import async_test


class Tracker:
    """Counts the jobs running at once and records the peak.

    A threading lock protects the counter, because the threading strategy runs
    every job in its own thread (and its own event loop).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def job(self, value, duration: float = 0.02):
        """Build a job that sleeps ``duration`` seconds and returns ``value``."""

        async def run():
            with self._lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
            try:
                await asyncio.sleep(duration)
                return value
            finally:
                with self._lock:
                    self.running -= 1

        return run


class Collector:
    """An ``on_result`` callback that stores (index, value) in arrival order."""

    def __init__(self) -> None:
        self.results = []

    def __call__(self, index, value) -> None:
        self.results.append((index, value))


class ConcurrencyStrategyTests(unittest.TestCase):
    """Behaviour shared by the asyncio and threading strategies."""

    strategies = (AsyncioStrategy, ThreadingStrategy)

    @async_test
    async def test_running_jobs_never_exceed_the_limit(self) -> None:
        """The peak number of simultaneous jobs is bounded by max_concurrency."""

        for strategy_class in self.strategies:
            with self.subTest(strategy=strategy_class.name):
                tracker = Tracker()
                collector = Collector()
                jobs = [tracker.job(i) for i in range(10)]

                await strategy_class(max_concurrency=3).run(jobs, collector)

                self.assertLessEqual(tracker.peak, 3)
                self.assertGreater(tracker.peak, 1)  # it really ran in parallel
                self.assertEqual(tracker.running, 0)
                self.assertEqual(len(collector.results), 10)

    @async_test
    async def test_sequential_runs_one_job_at_a_time(self) -> None:
        """SequentialStrategy never has more than one job running."""

        tracker = Tracker()
        collector = Collector()
        jobs = [tracker.job(i) for i in range(5)]

        await SequentialStrategy().run(jobs, collector)

        self.assertEqual(tracker.peak, 1)
        self.assertEqual(collector.results, [(i, i) for i in range(5)])

    @async_test
    async def test_results_arrive_in_completion_order_with_the_right_index(self) -> None:
        """Faster jobs report first, and each result carries its own index."""

        for strategy_class in self.strategies:
            with self.subTest(strategy=strategy_class.name):
                tracker = Tracker()
                collector = Collector()
                durations = [0.30, 0.05, 0.15]  # finishing order: 1, 2, 0
                jobs = [
                    tracker.job(f"job-{i}", duration)
                    for i, duration in enumerate(durations)
                ]

                await strategy_class(max_concurrency=3).run(jobs, collector)

                self.assertEqual(
                    collector.results,
                    [(1, "job-1"), (2, "job-2"), (0, "job-0")],
                )

    @async_test
    async def test_empty_list_works(self) -> None:
        """No jobs means no results and no error, for every strategy."""

        for strategy_class in (SequentialStrategy, *self.strategies):
            with self.subTest(strategy=strategy_class.name):
                collector = Collector()

                await strategy_class().run([], collector)

                self.assertEqual(collector.results, [])

    def test_limit_below_one_raises_value_error(self) -> None:
        """A max_concurrency below 1 is rejected."""

        for strategy_class in self.strategies:
            for limit in (0, -1):
                with self.subTest(strategy=strategy_class.name, limit=limit):
                    with self.assertRaises(ValueError):
                        strategy_class(max_concurrency=limit)
                    with self.assertRaises(ValueError):
                        create_strategy(strategy_class.name, limit)

    def test_unknown_strategy_name_raises_value_error(self) -> None:
        """create_strategy rejects names it does not know."""

        with self.assertRaises(ValueError):
            create_strategy("multiprocessing", 2)

    def test_create_strategy_builds_the_requested_class(self) -> None:
        """create_strategy maps each name to its strategy class."""

        for strategy_class in (SequentialStrategy, *self.strategies):
            with self.subTest(strategy=strategy_class.name):
                strategy = create_strategy(strategy_class.name, 2)
                self.assertIsInstance(strategy, strategy_class)


class AsyncioCancellationTests(unittest.TestCase):
    @async_test
    async def test_cancelling_the_task_leaves_no_running_job(self) -> None:
        """Cancelling run() cancels every job in flight and starts no new ones."""

        tracker = Tracker()
        collector = Collector()
        jobs = [tracker.job(i, duration=5) for i in range(6)]

        task = asyncio.create_task(
            AsyncioStrategy(max_concurrency=2).run(jobs, collector)
        )
        await asyncio.sleep(0.05)
        self.assertEqual(tracker.running, 2)  # the batch is really in progress

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(tracker.running, 0)
        self.assertEqual(collector.results, [])


