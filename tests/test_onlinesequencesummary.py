import pytest
from solver_comparison.logging.sequencesummary import OnlineSequenceSummary

testdata = [
    (1, range(1), [0]),
    (1, range(2), [0, 1]),
    (1, range(3), [0, 2]),
    (1, range(4), [0, 2, 3]),
    (1, range(5), [0, 4]),
    (1, range(6), [0, 4, 5]),
    (1, range(7), [0, 4, 6]),
    (1, range(8), [0, 4, 7]),
    (1, range(9), [0, 8]),
    (2, range(1), [0]),
    (2, range(2), [0, 1]),
    (2, range(3), [0, 1, 2]),
    (2, range(4), [0, 1, 2, 3]),
    (2, range(5), [0, 2, 4]),
    (2, range(6), [0, 2, 4, 5]),
    (2, range(7), [0, 2, 4, 6]),
    (2, range(8), [0, 2, 4, 7]),
    (2, range(9), [0, 4, 8]),
]


@pytest.mark.parametrize("n, inputs, expected_output", testdata)
def test_OnlineSequenceSummary(inputs, n, expected_output):
    summarizer = OnlineSequenceSummary(n=n)

    for input in inputs:
        summarizer.update(input)

    iterations, data = summarizer.get()
    assert data == expected_output
