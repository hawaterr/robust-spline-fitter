"""pytest auto-loads this file. It just wires up two CLI flags:

    pytest --plot-all         # save a plot for every test, pass or fail
    pytest --plot-failures    # save a plot only for tests that fail
    pytest                    # (default) no plots at all

This lives at the repo root, not in tests/, on purpose: pytest only honors
pytest_addoption from a conftest at rootdir. In tests/ the hook is silently
skipped and the flags come back as "unrecognized arguments".

You don't need to import anything from here -- request pytest's built-in
`request` fixture in a test function's arguments and read the flags off
`request.config.getoption(...)`.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--plot-all", action="store_true", default=False,
        help="Save a diagnostic plot for every test, pass or fail.",
    )
    parser.addoption(
        "--plot-failures", action="store_true", default=False,
        help="Save a diagnostic plot only for tests that fail.",
    )
