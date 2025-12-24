from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import ast
import re
from typing import Any, Iterable, Mapping, Optional

from tax_engine.data_containers.base_form import BaseForm

MISSING = object()


class MissingFieldError(Exception):
    """Raised when a required form field is missing during evaluation."""

    def __init__(self, field_name: str):
        super().__init__(f"Missing required field: {field_name}")
        self.field_name = field_name


def preprocess_expression(expression: str) -> str:
    """Normalize the P1 DSL into a Python-evaluable expression."""
    normalized = _replace_booleans(expression)
    normalized = normalized.replace("[]", "")
    normalized = _replace_implication(normalized)
    normalized = _replace_dots(normalized)
    return normalized


def _replace_booleans(expression: str) -> str:
    return re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", expression, flags=re.IGNORECASE), flags=re.IGNORECASE)


def _replace_implication(expression: str) -> str:
    if "=>" not in expression:
        return expression
    parts = [part.strip() for part in expression.split("=>")]
    result = parts[-1]
    for part in reversed(parts[:-1]):
        result = f"IMPLIES({part}, {result})"
    return result


def _replace_dots(expression: str) -> str:
    pattern = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
    updated = expression
    while True:
        updated, count = pattern.subn(r'DOT(\1, "\2")', updated)
        if count == 0:
            break
    return updated


def parse_expression(expression: str) -> ast.AST:
    """Parse a P1 DSL expression into a Python AST node."""
    normalized = preprocess_expression(expression)
    return ast.parse(normalized, mode="eval").body


@dataclass
class EvaluationContext:
    form: BaseForm
    item: Optional[Mapping[str, Any]] = None

    def get_form_value(self, name: str, allow_missing: bool) -> Any:
        if name == "CURRENT_YEAR":
            return date.today().year
        if name == "year":
            if self.form.year is not None:
                return self.form.year
        if name in self.form.fields:
            return self.form.fields[name]
        if allow_missing:
            return MISSING
        raise MissingFieldError(name)

    def get_item_value(self, name: str) -> Any:
        if self.item is None:
            return MISSING
        if name in self.item:
            return self.item[name]
        return MISSING


def to_decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal))


def round_decimal(value: Any, places: int) -> Any:
    decimal_value = to_decimal(value)
    if decimal_value is None:
        return MISSING
    quantizer = Decimal("1").scaleb(-places)
    return decimal_value.quantize(quantizer, rounding=ROUND_HALF_UP)


def dot_access(value: Any, attr: str) -> Any:
    if value is MISSING:
        return MISSING
    if isinstance(value, list):
        extracted = []
        for item in value:
            if isinstance(item, Mapping):
                extracted.append(item.get(attr, MISSING))
            else:
                extracted.append(getattr(item, attr, MISSING))
        return extracted
    if isinstance(value, Mapping):
        return value.get(attr, MISSING)
    return getattr(value, attr, MISSING)


def compare_with_tolerance(left: Any, right: Any, tolerance: Decimal, operator: ast.cmpop) -> bool:
    if left is MISSING or right is MISSING or left is None or right is None:
        return False
    if isinstance(left, list) and is_number(right):
        return _compare_list_to_number(left, right, tolerance, operator)
    if is_number(left) and isinstance(right, list):
        return _compare_list_to_number(right, left, tolerance, operator)
    if is_number(left) and is_number(right):
        left_decimal = to_decimal(left)
        right_decimal = to_decimal(right)
        if left_decimal is None or right_decimal is None:
            return False
        if isinstance(operator, ast.Eq):
            return abs(left_decimal - right_decimal) <= tolerance
        if isinstance(operator, ast.NotEq):
            return abs(left_decimal - right_decimal) > tolerance
        if isinstance(operator, ast.Lt):
            return left_decimal < right_decimal - tolerance
        if isinstance(operator, ast.LtE):
            return left_decimal <= right_decimal + tolerance
        if isinstance(operator, ast.Gt):
            return left_decimal > right_decimal + tolerance
        if isinstance(operator, ast.GtE):
            return left_decimal + tolerance >= right_decimal
    try:
        if isinstance(operator, ast.Eq):
            return left == right
        if isinstance(operator, ast.NotEq):
            return left != right
        if isinstance(operator, ast.Lt):
            return left < right
        if isinstance(operator, ast.LtE):
            return left <= right
        if isinstance(operator, ast.Gt):
            return left > right
        if isinstance(operator, ast.GtE):
            return left >= right
    except TypeError:
        return False
    return False


def _compare_list_to_number(values: list[Any], number: Any, tolerance: Decimal, operator: ast.cmpop) -> bool:
    if not values:
        return True
    for value in values:
        if value is MISSING or value is None:
            return False
        if not compare_with_tolerance(value, number, tolerance, operator):
            return False
    return True


class ExpressionEvaluator:
    """Evaluate P1 expressions safely against a form container."""

    def __init__(self, tolerance: Decimal):
        self.tolerance = tolerance

    def evaluate(self, node: ast.AST, context: EvaluationContext, allow_missing: bool = False) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if context.item is not None and node.id in context.item:
                return context.get_item_value(node.id)
            return context.get_form_value(node.id, allow_missing)
        if isinstance(node, ast.BinOp):
            left = self.evaluate(node.left, context)
            right = self.evaluate(node.right, context)
            if left is MISSING or right is MISSING or left is None or right is None:
                return MISSING
            left_decimal = to_decimal(left)
            right_decimal = to_decimal(right)
            if left_decimal is None or right_decimal is None:
                return MISSING
            if isinstance(node.op, ast.Add):
                return left_decimal + right_decimal
            if isinstance(node.op, ast.Sub):
                return left_decimal - right_decimal
            if isinstance(node.op, ast.Mult):
                return left_decimal * right_decimal
            if isinstance(node.op, ast.Div):
                if right_decimal == 0:
                    return MISSING
                return left_decimal / right_decimal
            return MISSING
        if isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand, context)
            if operand is MISSING or operand is None:
                return MISSING
            if isinstance(node.op, ast.USub) and is_number(operand):
                return -to_decimal(operand)
            if isinstance(node.op, ast.Not):
                return not bool(operand)
            return operand
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value_node in node.values:
                    value = self.evaluate(value_node, context, allow_missing=True)
                    if value in (False, MISSING, None):
                        return False
                return True
            if isinstance(node.op, ast.Or):
                for value_node in node.values:
                    value = self.evaluate(value_node, context, allow_missing=True)
                    if value not in (False, MISSING, None):
                        return True
                return False
        if isinstance(node, ast.Compare):
            left = self.evaluate(node.left, context)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self.evaluate(comparator, context)
                if not compare_with_tolerance(left, right, self.tolerance, operator):
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            func_name = self._resolve_func_name(node.func)
            if func_name in {"present", "if_present"}:
                value = self.evaluate(node.args[0], context, allow_missing=True)
                return value is not MISSING and value is not None
            if func_name == "allow_missing":
                _ = self.evaluate(node.args[0], context, allow_missing=True)
                return True
            if func_name == "matches_regex":
                value = self.evaluate(node.args[0], context, allow_missing=True)
                pattern = self.evaluate(node.args[1], context, allow_missing=True)
                if value in (MISSING, None) or pattern in (MISSING, None):
                    return False
                return re.fullmatch(str(pattern), str(value)) is not None
            if func_name == "is_integer":
                value = self.evaluate(node.args[0], context, allow_missing=True)
                if value in (MISSING, None):
                    return False
                if isinstance(value, int):
                    return True
                if isinstance(value, str):
                    return value.isdigit() or (value.startswith("-") and value[1:].isdigit())
                return False
            if func_name == "round":
                value = self.evaluate(node.args[0], context)
                places = self.evaluate(node.args[1], context)
                if value is MISSING or places is MISSING:
                    return MISSING
                return round_decimal(value, int(places))
            if func_name == "min":
                values = [self.evaluate(arg, context) for arg in node.args]
                if any(val is MISSING for val in values):
                    return MISSING
                return min(values)
            if func_name == "max":
                values = [self.evaluate(arg, context) for arg in node.args]
                if any(val is MISSING for val in values):
                    return MISSING
                return max(values)
            if func_name == "sum":
                values = self.evaluate(node.args[0], context)
                return self._sum_values(values)
            if func_name == "count":
                values = self.evaluate(node.args[0], context)
                if values is MISSING or values is None:
                    return MISSING
                if isinstance(values, list):
                    return len(values)
                return MISSING
            if func_name == "abs":
                value = self.evaluate(node.args[0], context)
                if value is MISSING or value is None:
                    return MISSING
                decimal_value = to_decimal(value)
                if decimal_value is None:
                    return MISSING
                return abs(decimal_value)
            if func_name == "for_each":
                values = self.evaluate(node.args[0], context)
                if values is MISSING or values is None:
                    return False
                if not isinstance(values, list):
                    return False
                expression = node.args[1]
                for item in values:
                    item_context = EvaluationContext(form=context.form, item=item if isinstance(item, Mapping) else {})
                    if not self.evaluate(expression, item_context, allow_missing=True):
                        return False
                return True
            if func_name == "DOT":
                value = self.evaluate(node.args[0], context)
                attr = self.evaluate(node.args[1], context)
                if value is MISSING or attr is MISSING:
                    return MISSING
                return dot_access(value, str(attr))
            if func_name == "IMPLIES":
                left = self.evaluate(node.args[0], context, allow_missing=True)
                if not left:
                    return True
                return bool(self.evaluate(node.args[1], context, allow_missing=True))
            if func_name == "all_money_fields_except":
                excluded = [self.evaluate(arg, context, allow_missing=True) for arg in node.args]
                values = context.form.get("all_money_fields", MISSING)
                if values is MISSING or values is None:
                    return MISSING
                if not isinstance(values, list):
                    return MISSING
                return [value for value in values if value not in excluded]
            raise ValueError(f"Unsupported function: {func_name}")
        if isinstance(node, ast.Subscript):
            value = self.evaluate(node.value, context)
            index = self.evaluate(node.slice, context)
            if value is MISSING or index is MISSING or value is None:
                return MISSING
            try:
                return value[index]
            except (KeyError, IndexError, TypeError):
                return MISSING
        if isinstance(node, ast.Index):
            return self.evaluate(node.value, context)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    def _resolve_func_name(self, func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        raise ValueError("Unsupported function expression")

    def _sum_values(self, values: Any) -> Any:
        if values is MISSING or values is None:
            return MISSING
        if not isinstance(values, list):
            return MISSING
        total = Decimal("0")
        for value in values:
            if value is MISSING or value is None:
                return MISSING
            decimal_value = to_decimal(value)
            if decimal_value is None:
                return MISSING
            total += decimal_value
        return total


def extract_required_fields(node: ast.AST) -> set[str]:
    """Collect referenced form field names required for evaluation."""
    required: set[str] = set()

    def visit(current: ast.AST, ignore: bool = False) -> None:
        if isinstance(current, ast.Name):
            if not ignore and current.id not in {"DOT", "IMPLIES", "CURRENT_YEAR"}:
                required.add(current.id)
            return
        if isinstance(current, ast.Call):
            func_name = current.func.id if isinstance(current.func, ast.Name) else ""
            if func_name in {"present", "if_present", "allow_missing"}:
                return
            if func_name == "all_money_fields_except":
                required.add("all_money_fields")
                return
            if func_name == "for_each":
                if current.args:
                    visit(current.args[0])
                return
            for arg in current.args:
                visit(arg)
            return
        for child in ast.iter_child_nodes(current):
            visit(child, ignore)

    visit(node)
    return {name for name in required if name not in {"True", "False"}}
