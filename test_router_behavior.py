#!/usr/bin/env python3
"""Deterministic behavioral regression tests for production llama-proxy.

Tests actual production functions and boundaries:
- llama_proxy.check_auth (fail closed on missing/empty key)
- llama_proxy.resolve_model (canonical Strata, retained aliases, rejected false aliases)
- llama_proxy.acquire_model (health cancel, real startup failure, startup cancellation lock retention, FIFO queued cancellation)
- llama_proxy.route_request (unknown model 404 before acquire, send cancel/failure release, network disconnect abort)
- llama_proxy.LeasedStreamingResponse (normal, pre-iteration disconnect, active stream disconnect)
"""
import asyncio
import importlib.util
import json
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from starlette.requests import Request

# Load production llama-proxy module
sys.path.insert(0, "/usr/local/bin")
spec = importlib.util.spec_from_file_location("llama_proxy", "/usr/local/bin/llama-proxy.py")
llama_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llama_proxy)


class TestAuthFailClosed(unittest.TestCase):
    def test_auth_missing_file_or_empty_key_fails_closed(self):
        """check_auth must reject (fail closed) when API key file is missing or empty."""
        with patch.object(llama_proxy, "get_api_key", return_value=""):
            req = MagicMock()
            req.headers = {"authorization": "Bearer any_token"}
            self.assertFalse(llama_proxy.check_auth(req), "Must fail closed if key is empty")

            req_empty = MagicMock()
            req_empty.headers = {}
            self.assertFalse(llama_proxy.check_auth(req_empty), "Must fail closed if no auth header")

    def test_auth_valid_and_invalid_tokens(self):
        """check_auth validates exact token against configured key."""
        with patch.object(llama_proxy, "get_api_key", return_value="valid-secret-token-xyz"):
            req_valid = MagicMock()
            req_valid.headers = {"authorization": "Bearer valid-secret-token-xyz"}
            self.assertTrue(llama_proxy.check_auth(req_valid))

            req_bad = MagicMock()
            req_bad.headers = {"authorization": "Bearer wrong-token"}
            self.assertFalse(llama_proxy.check_auth(req_bad))

            req_basic = MagicMock()
            req_basic.headers = {"authorization": "Basic valid-secret-token-xyz"}
            self.assertFalse(llama_proxy.check_auth(req_basic))


class TestModelResolutionAndAliases(unittest.TestCase):
    def setUp(self):
        self.mock_installed = [
            {"id": "Fara1.5-27B-Q4_K_M.gguf", "meta": {"path": "/models/fara.gguf"}},
            {"id": "Qwen3.8-27B-Q4_K_M.gguf", "meta": {"path": "/models/qwen27.gguf"}},
            {"id": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "meta": {"path": "/models/qwen35.gguf"}},
            {"id": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf", "meta": {"path": "/models/uncensored.gguf"}},
            {"id": "Qwen3.5-4B-UD-Q4_K_XL.gguf", "meta": {"path": "/models/qwen4.gguf"}},
        ]

    def test_canonical_strata_target(self):
        with patch.object(llama_proxy, "get_installed_gguf_models", return_value=self.mock_installed):
            for q in ["qwen3.8-flash-next", "openai/qwen3.8-flash-next", "Qwen3.8-Flash-Next"]:
                res = llama_proxy.resolve_model(q)
                self.assertIsNotNone(res)
                self.assertEqual(res["backend"], "strata")
                self.assertEqual(res["base_url"], "http://127.0.0.1:8090")
                self.assertEqual(res["id"], "qwen3.8-flash-next")

    def test_false_qwen35_35b_alias_removed(self):
        with patch.object(llama_proxy, "get_installed_gguf_models", return_value=self.mock_installed):
            res = llama_proxy.resolve_model("qwen3.5-35b")
            self.assertIsNone(res, "False alias 'qwen3.5-35b' must be rejected")

    def test_retained_aliases_resolution(self):
        with patch.object(llama_proxy, "get_installed_gguf_models", return_value=self.mock_installed):
            expected = {
                "fara1.5": "Fara1.5-27B-Q4_K_M.gguf",
                "qwen3.8-27b": "Qwen3.8-27B-Q4_K_M.gguf",
                "qwen3.6-35b": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                "qwen3.8-uncensored": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
                "qwen3.5-4b": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
            }
            for alias, expected_id in expected.items():
                res = llama_proxy.resolve_model(alias)
                self.assertIsNotNone(res, f"Failed for alias {alias}")
                self.assertEqual(res["backend"], "llama")
                self.assertEqual(res["id"], expected_id)


class TestAcquireModelProductionBehavior(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        llama_proxy.gpu_lock = asyncio.Lock()

    async def test_failed_start_keeps_lease_until_cleanup_finishes(self):
        cleanup_started = asyncio.Event()
        cleanup_allowed = asyncio.Event()

        async def cleanup(_backend):
            cleanup_started.set()
            await cleanup_allowed.wait()

        target = {"backend": "strata", "id": "qwen3.8-flash-next", "base_url": "http://127.0.0.1:8090"}
        with patch.object(llama_proxy, "check_backend_loaded_identity", return_value=False), \
             patch.object(llama_proxy, "switch_to_strata", side_effect=RuntimeError("allocation failed")), \
             patch.object(llama_proxy, "cleanup_backend_unit", side_effect=cleanup):
            task = asyncio.create_task(llama_proxy.acquire_model(target, MagicMock()))
            try:
                await cleanup_started.wait()
                for _ in range(5):
                    await asyncio.sleep(0)
                self.assertTrue(llama_proxy.gpu_lock.locked(), "A failed engine still owns the GPU during stop")
                self.assertFalse(task.done(), "Switch must not settle before its unit is stopped")
            finally:
                cleanup_allowed.set()
                await asyncio.gather(task, return_exceptions=True)
            self.assertFalse(llama_proxy.gpu_lock.locked())

    async def test_health_check_cancellation_releases_lock(self):
        """Cancellation during health check must not leak gpu_lock."""
        mock_client = MagicMock()
        target = {"backend": "llama", "id": "model.gguf", "base_url": "http://127.0.0.1:8081"}

        async def hanging_health(*args, **kwargs):
            await asyncio.sleep(10.0)
            return True

        with patch.object(llama_proxy, "check_backend_loaded_identity", side_effect=hanging_health):
            acquire_task = asyncio.create_task(llama_proxy.acquire_model(target, mock_client))
            await asyncio.sleep(0.01)

            # Task is waiting in health check, holding gpu_lock
            self.assertTrue(llama_proxy.gpu_lock.locked())

            # Cancel during health check
            acquire_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await acquire_task

            # Lock must be released!
            self.assertFalse(llama_proxy.gpu_lock.locked(), "Lock must be released on health check cancellation")

    async def test_startup_failure_cleans_up_and_releases_lock(self):
        """Startup failure in production acquire_model cleans up unit and releases gpu_lock."""
        mock_client = MagicMock()
        target = {"backend": "strata", "id": "qwen3.8-flash-next", "base_url": "http://127.0.0.1:8090"}
        cleanup_called = False

        async def mock_cleanup(backend):
            nonlocal cleanup_called
            cleanup_called = True

        async def failing_switch(*args, **kwargs):
            raise RuntimeError("Strata GPU VRAM allocation failed")

        with patch.object(llama_proxy, "check_backend_loaded_identity", return_value=False), \
             patch.object(llama_proxy, "switch_to_strata", side_effect=failing_switch), \
             patch.object(llama_proxy, "cleanup_backend_unit", side_effect=mock_cleanup):

            with self.assertRaises(RuntimeError):
                await llama_proxy.acquire_model(target, mock_client)

            self.assertTrue(cleanup_called, "cleanup_backend_unit must be called on failure")
            self.assertFalse(llama_proxy.gpu_lock.locked(), "Lock must be released after failed switch")

    async def test_startup_cancellation_retains_ownership_until_settled(self):
        """Client cancellation during startup retains GPU lock until startup task settles."""
        mock_client = MagicMock()
        target = {"backend": "llama", "id": "model.gguf", "base_url": "http://127.0.0.1:8081"}

        startup_settled = asyncio.Event()

        async def controlled_startup(*args, **kwargs):
            await startup_settled.wait()

        with patch.object(llama_proxy, "check_backend_loaded_identity", return_value=False), \
             patch.object(llama_proxy, "switch_to_llama", side_effect=controlled_startup):

            task1 = asyncio.create_task(llama_proxy.acquire_model(target, mock_client))
            await asyncio.sleep(0.01)

            # Task 1 holds lock and is executing startup
            self.assertTrue(llama_proxy.gpu_lock.locked())

            # Client 1 cancels
            task1.cancel()
            await asyncio.sleep(0.01)
            self.assertTrue(task1.cancelled())

            # GPU lock MUST still be held because startup task has not settled!
            self.assertTrue(llama_proxy.gpu_lock.locked(), "GPU lock must remain held while startup task is in flight")

            # Let startup settle
            startup_settled.set()
            await asyncio.sleep(0.02)

            # Now lock is released
            self.assertFalse(llama_proxy.gpu_lock.locked(), "GPU lock must be released once startup settles")

    async def test_fifo_queued_cancellation(self):
        """Cancelled queued request leaves FIFO without holding up following request."""
        mock_client = MagicMock()
        target = {"backend": "strata", "id": "qwen3.8-flash-next", "base_url": "http://127.0.0.1:8090"}

        with patch.object(llama_proxy, "check_backend_loaded_identity", return_value=True):
            # First acquire holds lock
            lease1 = await llama_proxy.acquire_model(target, mock_client)
            self.assertTrue(llama_proxy.gpu_lock.locked())

            order = []
            async def queue_client(name):
                try:
                    l = await llama_proxy.acquire_model(target, mock_client)
                    order.append(f"{name}_acquired")
                    l.release()
                except asyncio.CancelledError:
                    order.append(f"{name}_cancelled")
                    raise

            t_b = asyncio.create_task(queue_client("B"))
            t_c = asyncio.create_task(queue_client("C"))
            await asyncio.sleep(0.01)

            # Cancel B in queue
            t_b.cancel()
            await asyncio.sleep(0.01)

            # Release lease1
            lease1.release()

            await asyncio.gather(t_c, return_exceptions=True)
            await asyncio.gather(t_b, return_exceptions=True)

            self.assertEqual(order, ["B_cancelled", "C_acquired"])
            self.assertFalse(llama_proxy.gpu_lock.locked())


class TestRouteRequestBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        llama_proxy.gpu_lock = asyncio.Lock()
        llama_proxy.app.state.http_client = MagicMock()

    async def test_unknown_model_returns_404_before_acquire(self):
        """Unknown model returns 404 before acquiring GPU lock or attempting backend switch."""
        req = MagicMock()
        req.method = "POST"
        req.headers = {"authorization": "Bearer valid-key"}
        req.body = AsyncMock(return_value=b'{"model": "nonexistent-model-xyz"}')

        with patch.object(llama_proxy, "check_auth", return_value=True):
            resp = await llama_proxy.route_request(req, "v1/chat/completions")
            self.assertEqual(resp.status_code, 404)
            self.assertFalse(llama_proxy.gpu_lock.locked(), "Lock must not be acquired on 404")

    async def test_send_cancel_or_failure_releases_lease(self):
        """Failure or cancellation during upstream send releases the backend lease."""
        target = {"backend": "strata", "id": "qwen3.8-flash-next", "base_url": "http://127.0.0.1:8090"}
        receive_queue = asyncio.Queue()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", b"Bearer valid-key"), (b"content-type", b"application/json")],
            "query_string": b"",
        }

        async def mock_receive():
            return await receive_queue.get()

        await receive_queue.put({"type": "http.request", "body": b'{"model": "qwen3.8-flash-next"}', "more_body": False})
        req = Request(scope, mock_receive)

        mock_http_client = MagicMock()
        mock_http_client.build_request = MagicMock()
        mock_http_client.send = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        llama_proxy.app.state.http_client = mock_http_client

        with patch.object(llama_proxy, "check_auth", return_value=True), \
             patch.object(llama_proxy, "resolve_model", return_value=target), \
             patch.object(llama_proxy, "check_backend_loaded_identity", return_value=True):

            resp = await llama_proxy.route_request(req, "v1/chat/completions")
            self.assertEqual(resp.status_code, 502)
            self.assertFalse(llama_proxy.gpu_lock.locked(), "GPU lock/lease must be released on send failure")

    async def test_network_disconnect_cancels_in_flight_acquire(self):
        """When client TCP connection closes during queued acquire, handler cancels acquire immediately."""
        acquire_finished = False
        acquire_cancelled = False

        async def slow_acquire(*args, **kwargs):
            nonlocal acquire_finished, acquire_cancelled
            try:
                await asyncio.sleep(0.5)
                acquire_finished = True
                return llama_proxy.BackendLease("m", "http://127.0.0.1:8081", lambda: None)
            except asyncio.CancelledError:
                acquire_cancelled = True
                raise

        target = {"backend": "llama", "id": "model.gguf", "base_url": "http://127.0.0.1:8081"}
        receive_queue = asyncio.Queue()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", b"Bearer valid-key"), (b"content-type", b"application/json")],
            "query_string": b"",
        }

        async def mock_receive():
            return await receive_queue.get()

        await receive_queue.put({"type": "http.request", "body": b'{"model": "qwen3.8-27b"}', "more_body": False})
        req = Request(scope, mock_receive)

        with patch.object(llama_proxy, "check_auth", return_value=True), \
             patch.object(llama_proxy, "resolve_model", return_value=target), \
             patch.object(llama_proxy, "acquire_model", side_effect=slow_acquire):

            handler_task = asyncio.create_task(llama_proxy.route_request(req, "v1/chat/completions"))
            await asyncio.sleep(0.05)

            # Client disconnects via ASGI http.disconnect
            await receive_queue.put({"type": "http.disconnect"})

            try:
                await handler_task
            except asyncio.CancelledError:
                pass

            self.assertFalse(acquire_finished, "acquire must not have finished")
            self.assertTrue(acquire_cancelled, "acquire must have been cancelled by disconnect watcher")


    async def test_simultaneous_disconnect_and_upstream_headers_closes_lease(self):
        queue = asyncio.Queue()
        await queue.put({"type": "http.request", "body": b'{"model":"qwen3.5-4b"}', "more_body": False})
        await queue.put({"type": "http.disconnect"})
        request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                           "query_string": b"", "headers": []}, queue.get)

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"response"

        upstream = httpx.Response(200, stream=Stream())
        client = MagicMock()
        client.send = AsyncMock(return_value=upstream)
        llama_proxy.app.state.http_client = client
        target = {"backend": "llama", "id": "model.gguf", "base_url": "http://127.0.0.1:8081"}

        async def simultaneous(tasks, **_kwargs):
            await asyncio.gather(*tasks, return_exceptions=True)
            return set(tasks), set()

        with patch.object(llama_proxy, "check_auth", return_value=True), \
             patch.object(llama_proxy, "resolve_model", return_value=target), \
             patch.object(llama_proxy, "check_backend_loaded_identity", return_value=True), \
             patch.object(llama_proxy.asyncio, "wait", side_effect=simultaneous):
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await llama_proxy.route_request(request, "v1/chat/completions")
                self.assertTrue(upstream.is_closed, "Abandoned headers must not leave an upstream connection open")
                self.assertFalse(llama_proxy.gpu_lock.locked(), "Completed acquire must not lose its lease on disconnect")
            finally:
                await upstream.aclose()
                if llama_proxy.gpu_lock.locked():
                    llama_proxy.gpu_lock.release()

    async def test_handler_cancellation_reaps_pending_acquire(self):
        entered = asyncio.Event()
        inflight = None
        queue = asyncio.Queue()
        await queue.put({"type": "http.request", "body": b'{"model":"qwen3.5-4b"}', "more_body": False})
        request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                           "query_string": b"", "headers": []}, queue.get)

        async def health(*_args):
            nonlocal inflight
            inflight = asyncio.current_task()
            entered.set()
            await asyncio.Event().wait()

        target = {"backend": "llama", "id": "model.gguf", "base_url": "http://127.0.0.1:8081"}
        with patch.object(llama_proxy, "check_auth", return_value=True), \
             patch.object(llama_proxy, "resolve_model", return_value=target), \
             patch.object(llama_proxy, "check_backend_loaded_identity", side_effect=health):
            task = asyncio.create_task(llama_proxy.route_request(request, "v1/chat/completions"))
            try:
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(inflight.done(), "Handler cancellation must reap acquire before returning")
                self.assertFalse(llama_proxy.gpu_lock.locked())
            finally:
                if inflight and not inflight.done():
                    inflight.cancel()
                    await asyncio.gather(inflight, return_exceptions=True)


class TestLeasedStreamingResponseLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_response_call_lifecycle_normal(self):
        upstream_closed = False
        lease_released = False

        async def mock_aclose():
            nonlocal upstream_closed
            upstream_closed = True

        mock_upstream = MagicMock()
        mock_upstream.aclose = mock_aclose

        def mock_release():
            nonlocal lease_released
            lease_released = True

        lease = llama_proxy.BackendLease("m", "u", mock_release)

        async def dummy_generator():
            yield b"chunk1"
            yield b"chunk2"

        resp = llama_proxy.LeasedStreamingResponse(
            dummy_generator(),
            lease=lease,
            upstream=mock_upstream,
        )

        sent_messages = []
        stream_done = asyncio.Event()

        async def mock_send(msg):
            sent_messages.append(msg)
            if msg.get("type") == "http.response.body" and not msg.get("more_body", False):
                stream_done.set()

        async def mock_receive():
            await stream_done.wait()
            return {"type": "http.disconnect"}

        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}
        await resp(scope, mock_receive, mock_send)

        self.assertTrue(upstream_closed, "Upstream connection must be closed")
        self.assertTrue(lease_released, "Lease must be released")

    async def test_response_call_pre_iteration_and_active_stream_disconnect(self):
        """Lease must be released when client disconnects during active stream."""
        upstream_closed = False
        lease_released = False

        async def mock_aclose():
            nonlocal upstream_closed
            upstream_closed = True

        mock_upstream = MagicMock()
        mock_upstream.aclose = mock_aclose

        def mock_release():
            nonlocal lease_released
            lease_released = True

        lease = llama_proxy.BackendLease("m", "u", mock_release)

        async def dummy_generator():
            yield b"chunk1"

        resp = llama_proxy.LeasedStreamingResponse(
            dummy_generator(),
            lease=lease,
            upstream=mock_upstream,
        )

        async def failing_send(msg):
            if msg.get("type") == "http.response.body":
                raise ConnectionResetError("Client disconnected")

        async def mock_receive():
            return {"type": "http.disconnect"}

        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

        try:
            await resp(scope, mock_receive, failing_send)
        except (ConnectionResetError, BaseExceptionGroup):
            pass

        self.assertTrue(upstream_closed, "Upstream must be closed on disconnect")
        self.assertTrue(lease_released, "Lease must be released on disconnect")


if __name__ == "__main__":
    unittest.main()
