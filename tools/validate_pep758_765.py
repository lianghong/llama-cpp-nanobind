#!/usr/bin/env python3.14
"""Validate PEP 758 and PEP 765 compliance.

This script verifies that all Python code in the repository complies with:
- PEP 758: Unparenthesized exception lists (parentheses required with 'as')
- PEP 765: Control flow restrictions in finally blocks

Usage:
    python3.14 tools/validate_pep758_765.py

Exit codes:
    0: All files compliant
    1: Violations found or compilation errors
"""

import ast
import sys
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """Represents a PEP violation."""

    file: Path
    line: int
    pep: str
    message: str


def find_python_files() -> list[Path]:
    """Find all Python files in src/, tests/, examples/, tools/."""
    dirs = ["src", "tests", "examples", "tools"]
    files: list[Path] = []
    for d in dirs:
        path = Path(d)
        if path.exists():
            files.extend(path.rglob("*.py"))
    return sorted(files)


def check_pep_758(file_path: Path) -> list[Violation]:
    """Check PEP 758: exception lists with 'as' must be parenthesized.

    PEP 758 allows:
      except ValueError, TypeError:        # OK - no 'as'
      except (ValueError, TypeError) as e: # OK - parentheses with 'as'

    But disallows:
      except ValueError, TypeError as e:   # INVALID - 'as' requires parentheses
    """
    violations: list[Violation] = []
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        violations.append(
            Violation(
                file=file_path,
                line=e.lineno or 0,
                pep="PEP 758",
                message=f"Syntax error: {e.msg}",
            )
        )
        return violations

    # Note: ast.parse already enforces PEP 758 — invalid syntax won't parse.
    # This check is primarily for documentation/awareness. An ExceptHandler
    # that captures to a name AND has a Tuple of exception types necessarily
    # had parentheses in the source (otherwise it wouldn't have parsed).
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ExceptHandler)
            and node.name
            and node.type
            and isinstance(node.type, ast.Tuple)
        ):
            pass  # AST confirms parentheses were present — nothing to flag.

    return violations


def check_pep_765(file_path: Path) -> list[Violation]:
    """Check PEP 765: no return/break/continue exiting finally blocks.

    Disallowed patterns:
      try: ...
      finally: return/break/continue  # Direct exit from finally

    Allowed patterns:
      try: ...
      finally:
          def f(): return 42  # OK - exits nested function
          for x in y: break   # OK - exits nested loop
    """
    violations: list[Violation] = []
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        violations.append(
            Violation(
                file=file_path,
                line=e.lineno or 0,
                pep="PEP 765",
                message=f"Syntax error: {e.msg}",
            )
        )
        return violations

    class FinallyVisitor(ast.NodeVisitor):
        """Visit AST nodes to find control flow in finally blocks."""

        def __init__(self):
            self.in_finally_depth = 0
            self.violations: list[Violation] = []

        def visit_Try(self, node: ast.Try) -> None:
            # Visit try/except blocks normally
            for handler in node.handlers:
                self.visit(handler)
            for stmt in node.orelse:
                self.visit(stmt)

            # Enter finally block context
            if node.finalbody:
                self.in_finally_depth += 1
                for stmt in node.finalbody:
                    self.visit(stmt)
                self.in_finally_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Entering nested function - control flow is allowed inside
            if self.in_finally_depth > 0:
                # Inside finally but entering nested scope - reset depth
                old_depth = self.in_finally_depth
                self.in_finally_depth = 0
                self.generic_visit(node)
                self.in_finally_depth = old_depth
            else:
                self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            # Same as FunctionDef
            if self.in_finally_depth > 0:
                old_depth = self.in_finally_depth
                self.in_finally_depth = 0
                self.generic_visit(node)
                self.in_finally_depth = old_depth
            else:
                self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            # Entering nested loop - control flow is allowed inside
            if self.in_finally_depth > 0:
                old_depth = self.in_finally_depth
                self.in_finally_depth = 0
                self.generic_visit(node)
                self.in_finally_depth = old_depth
            else:
                self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            # Same as For
            if self.in_finally_depth > 0:
                old_depth = self.in_finally_depth
                self.in_finally_depth = 0
                self.generic_visit(node)
                self.in_finally_depth = old_depth
            else:
                self.generic_visit(node)

        def visit_Return(self, node: ast.Return) -> None:
            if self.in_finally_depth > 0:
                self.violations.append(
                    Violation(
                        file=file_path,
                        line=node.lineno,
                        pep="PEP 765",
                        message="return statement exits finally block",
                    )
                )
            self.generic_visit(node)

        def visit_Break(self, node: ast.Break) -> None:
            if self.in_finally_depth > 0:
                self.violations.append(
                    Violation(
                        file=file_path,
                        line=node.lineno,
                        pep="PEP 765",
                        message="break statement exits finally block",
                    )
                )
            self.generic_visit(node)

        def visit_Continue(self, node: ast.Continue) -> None:
            if self.in_finally_depth > 0:
                self.violations.append(
                    Violation(
                        file=file_path,
                        line=node.lineno,
                        pep="PEP 765",
                        message="continue statement exits finally block",
                    )
                )
            self.generic_visit(node)

    visitor = FinallyVisitor()
    visitor.visit(tree)
    return visitor.violations


def main() -> int:
    """Run validation and return exit code."""
    print("Validating PEP 758 and PEP 765 compliance...")
    print()

    files = find_python_files()
    print(f"Checking {len(files)} Python files...")
    print()

    all_violations: list[Violation] = []

    for file_path in files:
        violations = check_pep_758(file_path) + check_pep_765(file_path)
        all_violations.extend(violations)

    if all_violations:
        print("❌ VIOLATIONS FOUND:")
        print()
        for v in all_violations:
            print(f"{v.file}:{v.line}: {v.pep} - {v.message}")
        print()
        print(f"Total violations: {len(all_violations)}")
        return 1

    print("✅ All files compliant with PEP 758 and PEP 765")
    print()
    print("PEP 758: Unparenthesized exception lists (parentheses required with 'as')")
    print("PEP 765: No return/break/continue exiting finally blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
