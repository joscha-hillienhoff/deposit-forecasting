import pandas as pd
import pytest

from deposit_forecasting import synthetic


def test_deterministic_and_seed_dependent():
    a = synthetic.generate_synthetic_book(seed=3)
    b = synthetic.generate_synthetic_book(seed=3)
    c = synthetic.generate_synthetic_book(seed=4)
    pd.testing.assert_frame_equal(a.panel, b.panel)
    pd.testing.assert_series_equal(a.planned_outflows, b.planned_outflows)
    assert not a.panel.equals(c.panel)


def test_labelled_synthetic():
    book = synthetic.generate_synthetic_book()
    assert "SYNTHETIC" in book.label
    assert all(name.startswith("Synthetic Client ") for name in book.departures.index)
    assert "SYNTHETIC" in synthetic.demonstrate(book).attrs["label"]
    assert "SYNTHETIC" in (synthetic.__doc__ or "")


def test_schedule_matches_realised_balances():
    book = synthetic.generate_synthetic_book()
    assert book.panel["deposits"].equals(book.panel["retail"] + book.panel["large_clients"])
    drop = book.panel["large_clients"].diff().dropna()
    leaving = drop[drop < 0]
    assert (-leaving).to_dict() == pytest.approx(book.planned_outflows.to_dict())


def test_planned_outflows_improve_the_forecast():
    result = synthetic.demonstrate()
    err = lambda col: (result[col] - result["realised"]).abs().mean()  # noqa: E731
    assert err("drift_with_outflows") < err("drift")
