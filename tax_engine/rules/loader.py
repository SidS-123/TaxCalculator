from __future__ import annotations

from dataclasses import dataclass
import ast
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, List

from tax_engine.data_containers.base_form import BaseForm
from tax_engine.validators.result import ValidationResult
from tax_engine.validators.severity import Severity
from tax_engine.validators.validator import RuleFn

from .dsl import (
    MISSING,
    EvaluationContext,
    ExpressionEvaluator,
    MissingFieldError,
    compare_with_tolerance,
    extract_required_fields,
    parse_expression,
)


@dataclass(frozen=True)
class RuleDefinition:
    rule_id: str
    form_id: str
    description: str
    expression: str
    severity: Severity
    error_message: str
    tolerance: Decimal
    rule_type: str


def default_compendium_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "P1 Rules Compendium" / "P1.txt",
        repo_root / "P1_Rules_Compendium" / "P1.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("P1 Rules Compendium file not found.")


def load_rule_definitions(path: Path | None = None) -> List[RuleDefinition]:
    """Load rule definitions from the P1 Rules Compendium."""
    compendium_path = path or default_compendium_path()
    text = compendium_path.read_text(encoding="utf-8")
    forms = _parse_compendium(text)
    rules: List[RuleDefinition] = []
    for form_entry in forms:
        form_id = form_entry["form"].get("form_id", "")
        for rule_type in ("calculations", "constraints", "structural_rules"):
            for rule in form_entry.get(rule_type, []):
                rule_id = rule.get("id", "")
                description = rule.get("description", "")
                expression = rule.get("formula") or rule.get("rule") or ""
                severity = Severity(rule.get("severity", "ERROR"))
                error_message = rule.get("error_message", "")
                tolerance = Decimal(str(rule.get("tolerance", 0)))
                rules.append(
                    RuleDefinition(
                        rule_id=rule_id,
                        form_id=form_id,
                        description=description,
                        expression=expression,
                        severity=severity,
                        error_message=error_message,
                        tolerance=tolerance,
                        rule_type=rule_type,
                    )
                )
    return rules


def compile_rules(rules: Iterable[RuleDefinition]) -> dict[str, List[RuleFn]]:
    """Compile rule definitions into callable rule functions grouped by form_id."""
    registry: dict[str, List[RuleFn]] = {}
    for rule in rules:
        registry.setdefault(rule.form_id, []).append(_compile_rule(rule))
    return registry


def _compile_rule(rule: RuleDefinition) -> RuleFn:
    if rule.rule_type == "calculations":
        lhs_name, rhs_expression = _parse_formula(rule.expression)
        rhs_ast = parse_expression(rhs_expression)
        evaluator = ExpressionEvaluator(rule.tolerance)
        required_fields = extract_required_fields(rhs_ast) | {lhs_name}

        def rule_fn(form: BaseForm) -> ValidationResult:
            missing = _missing_fields(form, required_fields)
            if missing:
                return _skipped_result(rule, form, missing)
            try:
                context = EvaluationContext(form=form)
                rhs_value = evaluator.evaluate(rhs_ast, context)
                lhs_value = form.get(lhs_name)
                if rhs_value is MISSING or lhs_value is None:
                    return _skipped_result(rule, form, {lhs_name})
                passed = compare_with_tolerance(lhs_value, rhs_value, rule.tolerance, _eq_operator())
                message = rule.error_message if not passed else f"Passed: {rule.description}"
                return ValidationResult(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    passed=passed,
                    message=message,
                    form_id=form.form_id,
                )
            except MissingFieldError as exc:
                return _skipped_result(rule, form, {exc.field_name})
            except Exception as exc:  # pylint: disable=broad-except
                return ValidationResult(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    passed=False,
                    message=f"Rule evaluation error: {exc}",
                    form_id=form.form_id,
                )

        return rule_fn

    expression_ast = parse_expression(rule.expression)
    evaluator = ExpressionEvaluator(rule.tolerance)
    required_fields = extract_required_fields(expression_ast)

    def rule_fn(form: BaseForm) -> ValidationResult:
        missing = _missing_fields(form, required_fields)
        if missing:
            return _skipped_result(rule, form, missing)
        try:
            context = EvaluationContext(form=form)
            result = evaluator.evaluate(expression_ast, context, allow_missing=True)
            if result is MISSING:
                return _skipped_result(rule, form, required_fields)
            passed = bool(result)
            message = rule.error_message if not passed else f"Passed: {rule.description}"
            return ValidationResult(
                rule_id=rule.rule_id,
                severity=rule.severity,
                passed=passed,
                message=message,
                form_id=form.form_id,
            )
        except MissingFieldError as exc:
            return _skipped_result(rule, form, {exc.field_name})
        except Exception as exc:  # pylint: disable=broad-except
            return ValidationResult(
                rule_id=rule.rule_id,
                severity=rule.severity,
                passed=False,
                message=f"Rule evaluation error: {exc}",
                form_id=form.form_id,
            )

    return rule_fn


def _parse_formula(expression: str) -> tuple[str, str]:
    if "=" not in expression:
        raise ValueError(f"Calculation rule missing '=': {expression}")
    left, right = expression.split("=", 1)
    return left.strip(), right.strip()


def _missing_fields(form: BaseForm, required: Iterable[str]) -> set[str]:
    missing = set()
    for name in required:
        if name == "year":
            if form.year is None and not form.has(name):
                missing.add(name)
            continue
        if not form.has(name):
            missing.add(name)
    return missing


def _skipped_result(rule: RuleDefinition, form: BaseForm, missing: Iterable[str]) -> ValidationResult:
    missing_list = ", ".join(sorted(set(missing)))
    message = f"Skipped: missing fields [{missing_list}]"
    return ValidationResult(
        rule_id=rule.rule_id,
        severity=rule.severity,
        passed=True,
        message=message,
        form_id=form.form_id,
    )


def _parse_compendium(text: str) -> List[dict]:
    forms: List[dict] = []
    current_form: dict | None = None
    current_section: str | None = None
    current_rule: dict | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            if line.startswith("form:"):
                if current_form:
                    forms.append(current_form)
                current_form = {"form": {}, "calculations": [], "constraints": [], "structural_rules": []}
                current_section = "form"
                current_rule = None
                continue
            if line.startswith("calculations:"):
                current_section = "calculations"
                current_rule = None
                continue
            if line.startswith("constraints:"):
                current_section = "constraints"
                current_rule = None
                continue
            if line.startswith("structural_rules:"):
                current_section = "structural_rules"
                current_rule = None
                continue
            continue
        if current_form is None:
            continue
        stripped = line.strip()
        if current_section == "form":
            key, value = _parse_key_value(stripped)
            current_form["form"][key] = value
            continue
        if stripped.startswith("-"):
            current_rule = {}
            if current_section:
                current_form[current_section].append(current_rule)
            key, value = _parse_key_value(stripped[1:].strip())
            current_rule[key] = value
            continue
        if current_rule is not None:
            key, value = _parse_key_value(stripped)
            current_rule[key] = value

    if current_form:
        forms.append(current_form)
    return forms


def _parse_key_value(line: str) -> tuple[str, str | int | Decimal | bool | None]:
    if ":" not in line:
        return line, ""
    key, raw_value = line.split(":", 1)
    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"'):
        return key.strip(), value[1:-1]
    if value.lower() in {"true", "false"}:
        return key.strip(), value.lower() == "true"
    if value.lower() == "null":
        return key.strip(), None
    try:
        if "." in value:
            return key.strip(), Decimal(value)
        return key.strip(), int(value)
    except (ValueError, InvalidOperation):
        return key.strip(), value


def _eq_operator() -> ast.cmpop:
    return ast.Eq()
