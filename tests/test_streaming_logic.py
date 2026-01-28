"""Unit tests for streaming logic without requiring a model."""

import queue
import threading
import time


def test_queue_based_streaming_pattern():
    """Test that queue-based streaming pattern works as expected."""
    # This tests the same pattern used in generate_stream()
    # without needing an actual model

    token_queue: queue.Queue[int | None | Exception] = queue.Queue()

    def worker():
        """Simulated worker that generates tokens."""
        try:
            # Simulate generating 5 tokens with delay
            for i in range(5):
                time.sleep(0.01)  # Simulate generation time
                token_queue.put(i)
            token_queue.put(None)  # Sentinel
        except Exception as e:
            token_queue.put(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Collect results
    results = []
    start_time = time.time()
    first_item_time = None

    while True:
        item = token_queue.get()
        if first_item_time is None:
            first_item_time = time.time()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        results.append(item)

    end_time = time.time()
    thread.join(timeout=1.0)

    # Verify we got all tokens
    assert results == [0, 1, 2, 3, 4]

    # Verify first item arrived before all generation completed
    # (proves incremental, not buffered)
    time_to_first = first_item_time - start_time
    total_time = end_time - start_time

    # First item should arrive after ~1 token delay (0.01s)
    # Total time should be ~5 token delays (0.05s)
    # Ratio should be roughly 1/5 = 20%
    assert (
        time_to_first < total_time * 0.5
    ), "First item should arrive early (incremental)"


def test_exception_propagation():
    """Test that exceptions in worker thread propagate correctly."""
    token_queue: queue.Queue[int | None | Exception] = queue.Queue()

    def worker():
        """Worker that raises an exception."""
        try:
            token_queue.put(1)
            raise ValueError("Test exception")
        except Exception as e:
            token_queue.put(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # First item should be the token
    item = token_queue.get()
    assert item == 1

    # Second item should be the exception
    item = token_queue.get()
    assert isinstance(item, ValueError)
    assert str(item) == "Test exception"

    thread.join(timeout=1.0)


def test_early_termination():
    """Test that generator can stop early without hanging."""
    token_queue: queue.Queue[int | None] = queue.Queue()

    def worker():
        """Worker that generates many tokens."""
        for i in range(100):
            time.sleep(0.01)
            token_queue.put(i)
        token_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Take only first 3 tokens
    results = []
    for _ in range(3):
        item = token_queue.get()
        if item is not None:
            results.append(item)

    assert results == [0, 1, 2]

    # Thread is daemon, so it will be killed when test ends
    # This is the same behavior as generate_stream()


def test_streaming_performance():
    """Test that streaming provides tokens incrementally, not all at once."""
    token_queue: queue.Queue[int | None] = queue.Queue()
    timing_data = []

    def worker():
        """Worker that generates tokens with known delays."""
        for i in range(10):
            time.sleep(0.02)  # 20ms per token
            token_queue.put(i)
        token_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    start_time = time.time()
    thread.start()

    # Collect tokens and their arrival times
    while True:
        item = token_queue.get()
        if item is None:
            break
        timing_data.append(time.time() - start_time)

    thread.join(timeout=1.0)

    # Verify 10 tokens received
    assert len(timing_data) == 10

    # Verify incremental arrival (each token roughly 20ms apart)
    # If buffered, all would arrive at ~200ms
    # If incremental, first arrives at ~20ms, last at ~200ms
    assert timing_data[0] < 0.05, "First token should arrive quickly"
    assert timing_data[-1] > 0.15, "Last token should arrive after all generation"

    # Verify tokens arrived incrementally, not all at once
    # Check that there's spread in timing (not all at same time)
    time_spread = timing_data[-1] - timing_data[0]
    assert time_spread > 0.1, "Tokens should arrive over time, not all at once"
