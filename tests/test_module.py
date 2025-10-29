def add(a: int, b: int) -> int:
    return a + b


def test_add() -> None:
    assert add(2, 3) == 5  # noqa: PLR2004, S101
    assert add(0, 0) == 0  # noqa: S101
    assert add(-1, 1) == 0  # noqa: S101


def test_add_bis() -> None:
    assert add(2, 3) == 5  # noqa: PLR2004, S101
    assert add(0, 0) == 0  # noqa: S101
    assert add(-1, 1) == 0  # noqa: S101
