import re
import math
from typing import Dict, Any

def calculator_tool(expression: str) -> Dict[str, Any]:
    """
    Calculator Tool: Evaluates mathematical formulas and arithmetic expressions
    (e.g., additions, percentages, multiplications, powers, sqrt) safely.
    """
    # Replace common symbols
    clean_expr = expression.replace("x", "*").replace("X", "*").replace("^", "**")
    
    # Allow numbers, basic operators, parens, decimal points, and allowed math function names
    clean_expr = re.sub(r"[^0-9\+\-\*\/\%\.\(\)\s\w]", "", clean_expr).strip()

    if not clean_expr:
        return {"error": "Invalid or empty mathematical expression provided."}

    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e
    }

    try:
        # Validate that all words in expression are allowed function names
        tokens = re.findall(r"[a-zA-Z_]\w*", clean_expr)
        for t in tokens:
            if t not in allowed_names:
                return {"error": f"Unauthorized identifier in expression: '{t}'"}

        result = eval(clean_expr, {"__builtins__": None}, allowed_names)
        return {
            "expression": expression,
            "computed_result": float(result) if isinstance(result, (int, float)) else str(result),
            "summary": f"Calculated Result: {expression} = {result}"
        }
    except Exception as e:
        return {
            "error": f"Failed to compute expression '{expression}': {str(e)}"
        }
