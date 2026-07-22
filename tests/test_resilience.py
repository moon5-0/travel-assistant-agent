#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试、熔断与健康检查的离线测试。"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.circuit_breaker import CircuitBreaker, CircuitState
from utils.llm_resilience import (
    is_retriable_error,
    retry_with_backoff,
    run_health_check,
)


class TestRetryWithBackoff(unittest.IsolatedAsyncioTestCase):
    def test_only_temporary_errors_are_retriable(self):
        self.assertTrue(is_retriable_error(asyncio.TimeoutError()))
        self.assertTrue(is_retriable_error(RuntimeError("HTTP 503")))
        self.assertFalse(is_retriable_error(ValueError("invalid api key")))
        self.assertFalse(is_retriable_error(KeyError("missing field")))

    async def test_retries_with_exponential_delays_and_fresh_calls(self):
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise asyncio.TimeoutError("temporary timeout")
            return "ok"

        with patch("utils.llm_resilience.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await retry_with_backoff(
                operation,
                max_retries=2,
                base_delay_sec=1,
                max_delay_sec=30,
                jitter=False,
            )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1, 2])

    async def test_non_retriable_error_is_not_repeated(self):
        operation = AsyncMock(side_effect=ValueError("invalid request"))

        with patch("utils.llm_resilience.asyncio.sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(ValueError):
                await retry_with_backoff(operation, max_retries=3)

        self.assertEqual(operation.await_count, 1)
        sleep.assert_not_awaited()

    async def test_jitter_never_exceeds_configured_max_delay(self):
        operation = AsyncMock(side_effect=asyncio.TimeoutError("timeout"))

        with (
            patch("utils.llm_resilience.random.random", return_value=0.999),
            patch("utils.llm_resilience.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            with self.assertRaises(asyncio.TimeoutError):
                await retry_with_backoff(
                    operation,
                    max_retries=1,
                    base_delay_sec=30,
                    max_delay_sec=30,
                    jitter=True,
                )

        self.assertLessEqual(sleep.await_args.args[0], 30)


class TestCircuitBreaker(unittest.TestCase):
    def test_closed_open_half_open_and_recovery_flow(self):
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout_sec=10,
            half_open_successes=2,
        )

        with patch("utils.circuit_breaker.time.monotonic") as clock:
            clock.return_value = 100
            breaker.record_failure()
            breaker.record_failure()
            self.assertEqual(breaker.state, CircuitState.OPEN)
            self.assertFalse(breaker.allow_call())

            clock.return_value = 111
            self.assertEqual(breaker.state, CircuitState.HALF_OPEN)

        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)
        breaker.record_success()
        self.assertEqual(breaker.state, CircuitState.CLOSED)

    def test_half_open_failure_clears_probe_successes(self):
        breaker = CircuitBreaker(1, recovery_timeout_sec=0, half_open_successes=2)
        breaker.record_failure()
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)
        breaker.record_success()
        self.assertEqual(breaker._half_open_success_count, 1)

        breaker.record_failure()

        self.assertEqual(breaker._state, CircuitState.OPEN)
        self.assertEqual(breaker._half_open_success_count, 0)


class TestHealthCheck(unittest.IsolatedAsyncioTestCase):
    async def test_successful_empty_response_still_means_service_is_reachable(self):
        class FakeOpenAIChatModel:
            def __init__(self, **_kwargs):
                pass

            async def __call__(self, _messages):
                return types.SimpleNamespace(content="")

        model_module = types.ModuleType("agentscope.model")
        model_module.OpenAIChatModel = FakeOpenAIChatModel
        agentscope_module = types.ModuleType("agentscope")
        agentscope_module.model = model_module

        with patch.dict(
            sys.modules,
            {"agentscope": agentscope_module, "agentscope.model": model_module},
        ):
            ok, message = await run_health_check("url", "key", "model")

        self.assertTrue(ok)
        self.assertEqual(message, "ok")

    async def test_model_exception_marks_health_check_unavailable(self):
        class FailingOpenAIChatModel:
            def __init__(self, **_kwargs):
                pass

            async def __call__(self, _messages):
                raise ConnectionError("service unavailable")

        model_module = types.ModuleType("agentscope.model")
        model_module.OpenAIChatModel = FailingOpenAIChatModel
        agentscope_module = types.ModuleType("agentscope")
        agentscope_module.model = model_module

        with patch.dict(
            sys.modules,
            {"agentscope": agentscope_module, "agentscope.model": model_module},
        ):
            ok, message = await run_health_check("url", "key", "model")

        self.assertFalse(ok)
        self.assertIn("service unavailable", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
