from billing.invoice import invoice_total


def test_invoice_total():
    assert invoice_total(100) == 106.0


def test_zero():
    assert invoice_total(0) == 0
