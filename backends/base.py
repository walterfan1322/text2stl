"""Backend interface for 3D modeling engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BackendError(Exception):
    """Raised when a backend fails (import error, unknown name, exec failure)."""


class ModelBackend(ABC):
    """Abstract base class for a 3D modeling backend.

    A backend is responsible for:
    1. Providing a set of helper globals that are pre-injected into the
       LLM-generated code's execution environment.
    2. Executing LLM-generated code and producing an STL file.
    3. Declaring its allowed API surface (for the AST validator).
    4. Providing a system prompt snippet that describes its API to the LLM.
    """

    #: canonical name used in config ("trimesh" | "cadquery")
    name: str = ""

    @abstractmethod
    def helper_globals(self) -> dict:
        """Dict of pre-injected helpers available to generated code.

        Example: {"make_frustum": <callable>, ...}
        """

    @abstractmethod
    def execute_and_export(self, code: str, stl_path: Path) -> None:
        """Execute `code` and produce an STL at `stl_path`.

        Implementations must raise BackendError (or a subclass) on failure.
        """

    @abstractmethod
    def allowed_calls(self) -> set[str]:
        """Set of fully-qualified call names the backend considers valid.

        Used by the AST validator. Backend-specific methods (accessed through
        local variables or chained calls) are covered by the common
        ALLOWED_METHODS set in validators.py.
        """

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt describing this backend's API to the LLM."""

    @abstractmethod
    def enrich_prompt(self) -> str:
        """Return the design-specification prompt (enrichment step)."""

    @abstractmethod
    def review_prompt(self) -> str:
        """Return the code-review prompt for the auto-review step."""
