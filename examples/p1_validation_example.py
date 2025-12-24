from __future__ import annotations

from tax_engine.data_containers.base_form import BaseForm
from tax_engine.rules import RuleEngine


def main() -> None:
    form = BaseForm(form_id="W-2", year=2023)
    form.set("profile_id", "123")
    form.set("employer_name", "Example Corp")
    form.set("employer_id", "12-3456789")
    form.set("employee_name", "Jane Doe")
    form.set("ssn", "123-45-6789")
    form.set("wages", 50000)
    form.set("taxable_wages", 50000)
    form.set("all_money_fields", [50000, 6200, 725])

    engine = RuleEngine()
    summary = engine.validate_form(form)

    print(f"Form {summary.form_id} passed: {summary.passed}")
    print("Counts by severity:")
    for severity, count in summary.counts_by_severity.items():
        print(f"  {severity.value}: {count}")
    print("Results:")
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.rule_id} - {result.message}")


if __name__ == "__main__":
    main()
