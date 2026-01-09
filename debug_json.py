#!/usr/bin/env python3
import json
import sys

# Sample text that matches the actual LLM output format
text = """ {
    "action": "call_tool",
    "tool_name": "validate_minizinc",
    "arguments": {
        "mzn_code": "var 0..110: wheat;\\nvar 0..110: corn;\\nvar float: profit;\\nconstraint wheat + corn <= 110;\\nconstraint 3*wheat + 2*corn <= 240;\\nconstraint profit = 40*wheat + 30*corn;\\nsolve maximize profit;\\noutput [\\"wheat=\\", show(wheat), \\" corn=\\", show(corn), \\" profit=\\", show(profit)];"
    },
    "summary": "Validating the MiniZinc model syntax"
}"""

def extract_json_plans(text: str) -> list[dict]:
    """Extract all JSON objects from a text blob."""
    items = []
    buf = []
    depth = 0
    in_string = False
    escape_next = False
    
    for ch in text:
        if ch == '{' and not in_string:
            depth += 1
            buf.append(ch)
        elif ch == '}' and not in_string:
            buf.append(ch)
            depth -= 1
            if depth == 0 and buf:
                candidate = ''.join(buf)
                buf.clear()
                # Try standard parse first
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        items.append(obj)
                        continue
                except json.JSONDecodeError as e:
                    print(f"Standard parse failed: {e}", file=sys.stderr)
                # Fix literal newlines/tabs in string values
                try:
                    fixed = candidate.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        items.append(obj)
                        print("Fixed parse succeeded", file=sys.stderr)
                except json.JSONDecodeError as e:
                    print(f"Fixed parse also failed: {e}", file=sys.stderr)
        elif depth > 0:
            if ch == '"' and not escape_next:
                in_string = not in_string
            escape_next = (ch == '\\' and not escape_next)
            buf.append(ch)
    return items

plans = extract_json_plans(text)
print(f"Found {len(plans)} plans")
if plans:
    print(f"Action: {plans[0].get('action')}")
    print(f"Tool: {plans[0].get('tool_name')}")
