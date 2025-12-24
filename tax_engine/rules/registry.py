from __future__ import annotations

from pathlib import Path
from typing import List

from tax_engine.validators.validator import RuleFn

from .loader import compile_rules, load_rule_definitions


class RuleRegistry:
    """Registry of compiled P1 rules grouped by form_id."""

    def __init__(self, compendium_path: Path | None = None):
        self._rules = compile_rules(load_rule_definitions(compendium_path))

    def get_rules_for_form(self, form_id: str) -> List[RuleFn]:
        """Return all compiled rule functions for a given form_id."""
        return list(self._rules.get(form_id, []))


_default_registry: RuleRegistry | None = None


def get_rules_for_form(form_id: str, compendium_path: Path | None = None) -> List[RuleFn]:
    """Convenience access to registry rules without managing a registry instance."""
    global _default_registry
    if _default_registry is None or compendium_path is not None:
        _default_registry = RuleRegistry(compendium_path)
    return _default_registry.get_rules_for_form(form_id)
