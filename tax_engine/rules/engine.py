from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from tax_engine.data_containers.base_form import BaseForm
from tax_engine.validators.result import ValidationResult
from tax_engine.validators.severity import Severity

from .registry import RuleRegistry


@dataclass(frozen=True)
class ValidationSummary:
    """Aggregate results for a form validation run."""

    form_id: str
    passed: bool
    results: List[ValidationResult]
    counts_by_severity: Dict[Severity, int]


class RuleEngine:
    """Execute compiled P1 rules against a form container."""

    def __init__(self, registry: RuleRegistry | None = None):
        self._registry = registry or RuleRegistry()

    def validate_form(self, form: BaseForm) -> ValidationSummary:
        rules = self._registry.get_rules_for_form(form.form_id)
        results = [rule(form) for rule in rules]
        ordered = self._order_results(results)
        counts = self._count_by_severity(ordered)
        passed = all(result.passed for result in ordered)
        return ValidationSummary(
            form_id=form.form_id,
            passed=passed,
            results=ordered,
            counts_by_severity=counts,
        )

    @staticmethod
    def _order_results(results: List[ValidationResult]) -> List[ValidationResult]:
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.ERROR: 1,
            Severity.WARNING: 2,
            Severity.INFO: 3,
        }
        return sorted(results, key=lambda result: severity_order.get(result.severity, 99))

    @staticmethod
    def _count_by_severity(results: List[ValidationResult]) -> Dict[Severity, int]:
        counts: Dict[Severity, int] = {severity: 0 for severity in Severity}
        for result in results:
            counts[result.severity] += 1
        return counts
