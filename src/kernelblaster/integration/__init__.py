try:
    from .kgen_cudacoder import run_kgen_pipeline
except Exception:
    run_kgen_pipeline = None

__all__ = ["run_kgen_pipeline"]
