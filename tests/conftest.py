"""
Pytest configuration for matcher evaluation helpers.
"""


def pytest_addoption(parser):
    """Add optional CLI flags used by matcher tests."""
    parser.addoption(
        "--use-llm",
        action="store_true",
        default=False,
        help="Enable OpenAI-based matcher accuracy checks in tests",
    )


def pytest_configure(config):
    """Expose custom options for easier access in tests."""
    config.addinivalue_line(
        "markers",
        "matchers: marker for matcher-related tests",
    )


def pytest_collection_modifyitems(config, items):
    """Tag matcher tests with a marker for easier filtering."""
    for item in items:
        if "test_matcher_accuracy" in item.nodeid:
            item.add_marker("matchers")

