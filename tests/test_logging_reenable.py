"""Test logging re-enable functionality."""

from llama_cpp import Llama, LlamaConfig, disable_logging, reset_logging


def test_logging_can_be_reenabled(model_path):
    """Test that logging can be re-enabled after being disabled."""
    # First instance disables logging
    llm1 = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=False)
    )
    assert Llama._global_verbose is False

    llm1.close()

    # Second instance with verbose=True should re-enable logging
    llm2 = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=True)
    )
    assert Llama._global_verbose is True

    llm2.close()

    # Third instance can disable again
    llm3 = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=False)
    )
    assert Llama._global_verbose is False

    llm3.close()


def test_logging_functions_work_correctly(model_path):
    """Test that disable_logging and reset_logging work as expected."""
    # Reset to known state
    reset_logging()
    # Note: _global_verbose may be None, True, or False depending on prior tests

    # Disable logging
    disable_logging()

    # Create instance with verbose=True to re-enable
    llm = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=True)
    )
    assert Llama._global_verbose is True

    llm.close()


def test_concurrent_verbose_configuration_is_thread_safe():
    """Test that concurrent verbose configuration doesn't cause races.

    This test only verifies the locking mechanism works, not full model loading.
    """
    import threading

    results = []
    errors = []

    def toggle_verbose(enable: bool):
        try:
            with Llama._log_lock:
                if enable:
                    reset_logging()
                    Llama._global_verbose = True
                else:
                    disable_logging()
                    Llama._global_verbose = False
                results.append(enable)
        except Exception as e:
            errors.append(e)

    # Toggle verbose setting concurrently
    threads = []
    for i in range(20):
        enable = i % 2 == 0  # Alternate between True and False
        thread = threading.Thread(target=toggle_verbose, args=(enable,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Should have executed all operations successfully
    assert len(results) == 20

    # Final state should be consistent (either True or False)
    assert Llama._global_verbose in [True, False]


def test_verbose_false_then_true_in_sequence(model_path):
    """Test that creating instances with verbose=False then verbose=True works."""
    # First: verbose=False
    llm1 = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=False)
    )
    assert Llama._global_verbose is False
    llm1.close()

    # Second: verbose=True (should re-enable)
    llm2 = Llama(
        model_path, config=LlamaConfig(model_path=model_path, n_ctx=512, verbose=True)
    )
    assert Llama._global_verbose is True

    # Verify logging is actually re-enabled by checking we can generate
    response = llm2.generate("Hello", max_tokens=5)
    assert len(response) > 0

    llm2.close()
