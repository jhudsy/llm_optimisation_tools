#!/bin/bash
# Test script to capture MCP server startup errors

cd /Users/csc342/scratch/optimise_mcp

echo "Testing MCP server startup..." >&2
echo "Working directory: $(pwd)" >&2
echo "Python: /Users/csc342/scratch/optimise_mcp/.venv/bin/python3" >&2
echo "" >&2

# Try to start the server and capture stderr
timeout 3s /Users/csc342/scratch/optimise_mcp/.venv/bin/python3 -m src.modeller_checker.mcp --stdio 2>&1 <<EOF
{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1.0.0", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}
EOF

echo "" >&2
echo "Exit code: $?" >&2
