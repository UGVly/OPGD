# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Math-Verify scoring with optional process-isolated hard timeouts.

Math-Verify's built-in POSIX timeout uses ``SIGALRM``.  A pathological SymPy
operation can remain inside native code without returning control to Python,
so the signal timeout is not a hard process boundary.  OPGD reference runs
opt into the isolated path through ``VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS``.
Ordinary callers retain the original in-process behavior when that variable is
unset or non-positive.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import threading
from multiprocessing.connection import Connection
from typing import Optional

try:
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


logger = logging.getLogger(__name__)

_score_worker_lock = threading.Lock()
_score_worker_process: Optional[multiprocessing.Process] = None
_score_worker_connection: Optional[Connection] = None
_score_request_id = 0


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _compute_score_direct(model_output: str, ground_truth: str) -> float:
    """Preserve the original scorer exactly for successful evaluations."""

    result = False
    answer = remove_boxed(last_boxed_only_string(model_output))
    if answer is None:
        return 0.0

    try:
        if len(answer) > 300:
            answer = answer[:300]
        result = verify(parse("\\boxed{" + ground_truth + "}"), parse("\\boxed{" + answer + "}"))
    except Exception:
        pass

    return 1.0 if result else 0.0


def _score_worker_main(connection: Connection) -> None:
    """Evaluate requests serially in a disposable spawned process."""

    try:
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if request is None:
                return

            request_id, model_output, ground_truth = request
            try:
                score = _compute_score_direct(model_output, ground_truth)
            except BaseException:
                # The historical scorer maps all scoring failures to zero.  A
                # BaseException must not silently kill the isolation worker and
                # leave the parent waiting on a pipe forever.
                score = 0.0

            try:
                connection.send((request_id, score))
            except (BrokenPipeError, EOFError, OSError):
                return
    finally:
        connection.close()


def _shutdown_score_worker() -> None:
    """Close and reap the current isolation worker, escalating if hung."""

    global _score_worker_connection, _score_worker_process

    connection = _score_worker_connection
    process = _score_worker_process
    _score_worker_connection = None
    _score_worker_process = None

    if connection is not None:
        connection.close()
    if process is None:
        return

    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    else:
        process.join(timeout=0)


def _ensure_score_worker() -> tuple[multiprocessing.Process, Connection]:
    """Return a live spawn-based worker without inheriting the CUDA runtime."""

    global _score_worker_connection, _score_worker_process

    if (
        _score_worker_process is not None
        and _score_worker_connection is not None
        and _score_worker_process.is_alive()
    ):
        return _score_worker_process, _score_worker_connection

    _shutdown_score_worker()
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(target=_score_worker_main, args=(child_connection,), daemon=True)
    process.start()
    child_connection.close()
    _score_worker_process = process
    _score_worker_connection = parent_connection
    return process, parent_connection


def _hard_timeout_seconds() -> float:
    raw_value = os.environ.get("VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS", "0")
    try:
        value = float(raw_value)
    except ValueError as err:
        raise ValueError(
            "VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS must be a non-negative number"
        ) from err
    if value < 0:
        raise ValueError("VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS must be non-negative")
    return value


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    """Score one answer, optionally enforcing an external hard timeout.

    The spawned process is persistent, so normal requests pay IPC cost without
    repeatedly importing SymPy.  A timeout or broken worker is discarded before
    the next request, preventing stale replies and poisoned parser state.
    """

    timeout_seconds = _hard_timeout_seconds()
    timeout_score = float(timeout_score)
    if timeout_seconds == 0:
        return _compute_score_direct(model_output, ground_truth)

    global _score_request_id
    with _score_worker_lock:
        process, connection = _ensure_score_worker()
        _score_request_id += 1
        request_id = _score_request_id

        try:
            connection.send((request_id, model_output, ground_truth))
        except (BrokenPipeError, EOFError, OSError):
            _shutdown_score_worker()
            logger.warning(
                "math_verify_hard_timeout_event=worker_send_failure timeout_score=%s", timeout_score
            )
            return timeout_score

        if not connection.poll(timeout_seconds):
            worker_pid = process.pid
            _shutdown_score_worker()
            logger.warning(
                "math_verify_hard_timeout_event=deadline timeout_seconds=%.3f worker_pid=%s timeout_score=%s",
                timeout_seconds,
                worker_pid,
                timeout_score,
            )
            return timeout_score

        try:
            response_id, score = connection.recv()
        except (EOFError, OSError):
            _shutdown_score_worker()
            logger.warning(
                "math_verify_hard_timeout_event=worker_exit timeout_score=%s", timeout_score
            )
            return timeout_score

        if response_id != request_id:
            _shutdown_score_worker()
            logger.warning(
                "math_verify_hard_timeout_event=protocol_mismatch timeout_score=%s", timeout_score
            )
            return timeout_score
        return score


atexit.register(_shutdown_score_worker)
