"""Safe arithmetic calculator without eval()."""

import ast
import operator

from pydantic import BaseModel

from agent_research.tools.registry import ToolError


class CalculatorInput(BaseModel):
    expression: str


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("Exponent is too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ToolError("Expression contains unsupported syntax")


def calculator(inputs: CalculatorInput) -> int | float:
    if len(inputs.expression) > 200:
        raise ToolError("Expression is too long")
    try:
        return _evaluate(ast.parse(inputs.expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ToolError(str(exc)) from exc
