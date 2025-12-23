from __future__ import annotations
from typing import Callable, List
from tax_engine.data_containers.base_form import BaseForm
from .result import ValidationResult

RuleFn = Callable[[BaseForm], ValidationResult]

class Validator:
    def __init__(self, rules: List[RuleFn]):
        self.rules = rules

    def validate(self, form: BaseForm) -> List[ValidationResult]:
        results: List[ValidationResult] = []
        for rule_fn in self.rules:
            results.append(rule_fn(form))
        return results
