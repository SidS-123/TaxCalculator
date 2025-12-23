from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class BaseForm:
    """
    Generic container for extracted values.
    - `fields` holds raw extracted variables (strings, numbers, dates, booleans, lists, etc.)
    - Missing fields are allowed (rules can skip until those fields exist).
    """
    form_id: str
    year: Optional[int] = None
    fields: Dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def has(self, name: str) -> bool:
        return name in self.fields

    def set(self, name: str, value: Any) -> None:
        self.fields[name] = value
