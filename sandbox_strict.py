"""RestrictedPython sandbox layer (S7.3).

Adds a second-line defence on top of the existing AST allowlist
(validators.py). Where the allowlist refuses any import or call name
not on the list, this layer refuses the dangerous *behaviour* even
when an attacker has crafted code that passes static checks.

Strategy:
- Use RestrictedPython.compile_restricted_exec to compile the cleaned
  user code with restricted builtins.
- Provide a minimal whitelisted builtin set (range, enumerate, len,
  zip, abs, max, min, round, sum, sorted, reversed, list, dict, set,
  tuple, str, int, float, bool, isinstance, type, print).
- Forbid open(), __import__, getattr/setattr/delattr (RestrictedPython
  controls these via guarded equivalents).
- Only enable when config.sandbox_strict = true. Default off — we
  let the existing AST allowlist do its job until we've shadow-tested.

Test Strategy:
- shadow-mode: run both lenient and strict for a week, compare exec_ok
  rates. If strict has > 2pp regression, find the culprits, add to
  allowlist.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("text2stl.sandbox_strict")

# Cache the compile function so we don't repeatedly check imports
_compile_restricted: Callable | None = None
_safe_builtins: dict | None = None


def _get_restricted_python():
    """Lazy import; return (compile_restricted_exec, safe_builtins) or None."""
    global _compile_restricted, _safe_builtins
    if _compile_restricted is not None:
        return _compile_restricted, _safe_builtins
    try:
        from RestrictedPython import compile_restricted_exec
        from RestrictedPython.Guards import safe_builtins, safer_getattr
        from RestrictedPython.Eval import default_guarded_getitem
        # Augment safe_builtins with the things our generated code needs.
        # CadQuery + numpy heavy code uses: range, enumerate, len, tuple,
        # list, dict, isinstance, type, abs, max, min, round, sum, zip,
        # reversed, sorted, set, frozenset, bool, int, float, str.
        b = dict(safe_builtins)
        for name in ("range", "enumerate", "len", "tuple", "list", "dict",
                     "isinstance", "type", "abs", "max", "min", "round",
                     "sum", "zip", "reversed", "sorted", "set", "frozenset",
                     "bool", "int", "float", "str", "bytes", "print",
                     "iter", "next", "any", "all", "map", "filter",
                     "object", "Exception", "ValueError", "TypeError",
                     "RuntimeError", "AttributeError", "IndexError",
                     "KeyError", "ZeroDivisionError"):
            try:
                import builtins as _bi
                if hasattr(_bi, name):
                    b[name] = getattr(_bi, name)
            except Exception:
                pass
        _safe_builtins = b
        _compile_restricted = compile_restricted_exec
        return _compile_restricted, _safe_builtins
    except ImportError:
        log.info("RestrictedPython not installed — sandbox_strict disabled")
        return None, None


def is_available() -> bool:
    fn, _ = _get_restricted_python()
    return fn is not None


def compile_strict(code: str, filename: str = "<gen>") -> tuple[Any, list[str]]:
    """Compile under RestrictedPython.

    Returns (code_object, errors). On error, code_object is None.
    """
    fn, _ = _get_restricted_python()
    if fn is None:
        return None, ["RestrictedPython not available"]
    res = fn(code, filename=filename)
    code_obj = getattr(res, "code", None) or res[0] if isinstance(res, tuple) else getattr(res, "code", None)
    errors = list(getattr(res, "errors", []) or [])
    if code_obj is None:
        return None, errors or ["compile_restricted produced no code"]
    return code_obj, errors


def exec_strict(code: str, helper_globals: dict, output_path: str) -> dict:
    """Execute `code` under restricted builtins. Returns the exec globals.

    Raises RuntimeError if RestrictedPython is unavailable or compile
    failed, or whatever the code raised.
    """
    fn, builtins = _get_restricted_python()
    if fn is None:
        raise RuntimeError("RestrictedPython not available")
    code_obj, errors = compile_strict(code)
    if code_obj is None:
        raise RuntimeError("Restricted compile failed: " + "; ".join(errors))

    g: dict[str, Any] = {
        "__builtins__": builtins,
        "OUTPUT_PATH": output_path,
    }
    g.update(helper_globals)
    # RestrictedPython expects these guarded helpers in the namespace
    try:
        from RestrictedPython.Guards import (
            safer_getattr, guarded_iter_unpack_sequence,
            guarded_unpack_sequence,
        )
        from RestrictedPython.Eval import (
            default_guarded_getitem, default_guarded_getiter,
        )
        g["_getattr_"] = safer_getattr
        g["_getitem_"] = default_guarded_getitem
        g["_getiter_"] = default_guarded_getiter
        g["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        g["_unpack_sequence_"] = guarded_unpack_sequence
        # write guard is permissive — generated code legitimately
        # mutates Workplane objects. We're not protecting against
        # in-process state here, just against IO/import attacks.
        def _write(x):
            return x
        g["_write_"] = _write
    except ImportError:
        pass

    exec(code_obj, g)
    return g
