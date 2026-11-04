try:
    # Python ≥3.8
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # fallback for older Python
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("psi")  # must match [project] name
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback if package not installed