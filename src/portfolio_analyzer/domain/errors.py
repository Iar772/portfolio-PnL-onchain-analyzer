"""Domain-level failures independent of Ethereum and external providers."""


class DomainError(Exception):
    """Represent an invalid financial operation or accounting state."""


class InsufficientQuantityError(DomainError):
    """Reject a FIFO disposal that exceeds available inventory."""
