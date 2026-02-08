import clang.cindex
import sys
import json
from .rules import RULES

def analyze_file(file_path):
    # Mac users might need to set library path manually in env vars
    index = clang.cindex.Index.create()
    tu = index.parse(file_path, args=['-x', 'cuda', '--cuda-gpu-arch=sm_50'])
    
    issues = []

    def walk_ast(node):
        # Check for Hardcoded 32
        if node.kind == clang.cindex.CursorKind.INTEGER_LITERAL:
            tokens = list(node.get_tokens())
            if tokens and tokens[0].spelling == '32':
                issues.append({
                    "line": node.location.line,
                    "rule": RULES["WARP_SIZE"]
                })
        
        # Check for Call Expressions (Intrinsics)
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
            if "__shfl" in node.spelling:
                issues.append({
                    "line": node.location.line,
                    "rule": RULES["SHUFFLE_SYNC"]
                })

        for child in node.get_children():
            walk_ast(child)

    walk_ast(tu.cursor)
    return issues

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(json.dumps(analyze_file(sys.argv[1]), indent=2))
