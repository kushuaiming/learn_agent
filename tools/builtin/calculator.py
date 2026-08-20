import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool
from core.exceptions import ToolException


class CalculatorTool(Tool):
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    def __init__(self):
        super().__init__(
            name="python_calculator",
            description="Execute mathematical calculations. Supports basic arithmetic and math functions. For example: 2+3*4, sqrt(16), sin(pi/2), etc.",
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        Execute the calculation.

        Args:
            parameters: Dictionary containing the input parameter.

        Returns:
            Calculation result.
        """
        # Support two parameter formats: input and expression
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "Error: calculation expression cannot be empty"

        print(f"Calculating: {expression}")

        try:
            # Parse the expression
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body)
            result_str = str(result)
            print(f"Calculate Result: {result_str}")
            return result_str
        except Exception as e:
            error_msg = f"Failed to calculate: {str(e)}"
            print(f"Error Msg: {error_msg}")
            return error_msg

    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left), self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"Undefined variable: {node.id}")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")

    def get_parameters(self):
        """Get tool parameter definitions."""
        from ..base import ToolParameter

        return [
            ToolParameter(
                name="input",
                type="string",
                description="The mathematical expression to calculate, supports basic arithmetic and math functions",
                required=True,
            )
        ]


def calculate(expression: str) -> str:
    """
    Execute a mathematical calculation.

    Args:
        expression: The mathematical expression.

    Returns:
        Calculation result as a string.
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})
