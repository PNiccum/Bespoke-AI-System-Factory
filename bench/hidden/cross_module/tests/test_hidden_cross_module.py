from billing.invoice import invoice_total


def test_other_amounts():
    assert abs(invoice_total(50) - 53.0) < 1e-9
    assert abs(invoice_total(19.99) - 21.19) < 1e-9
    assert type(invoice_total(10)) is float


def test_rounded_to_cents():
    assert invoice_total(0.15) == round(invoice_total(0.15), 2)
    assert abs(invoice_total(0.15) - 0.16) < 1e-9
