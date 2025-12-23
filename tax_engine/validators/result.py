from __future__ import annotations
from dataclasses import dataclass
from .severity import Severity

@dataclass(frozen=True)
class ValidationResult:
    rule_id: str
    severity: Severity
    passed: bool
    message: str
    form_id: str
