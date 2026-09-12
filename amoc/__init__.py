__all__ = [
    "AgeAwareAMoCEngine",
    "AMoCv4",
]


def __getattr__(name):
    if name in __all__:
        from amoc.pipeline import AgeAwareAMoCEngine, AMoCv4

        return {"AgeAwareAMoCEngine": AgeAwareAMoCEngine, "AMoCv4": AMoCv4}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
