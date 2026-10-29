from billing.rates import TAX_RATE


def invoice_total(subtotal):
    """Subtotal plus sales tax, rounded to cents."""
    return round(subtotal * (1 + TAX_RATE), 2)
