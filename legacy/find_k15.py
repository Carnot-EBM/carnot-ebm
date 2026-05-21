import ast

with open("python/carnot/pipeline/verify_repair.py") as f:
    tree = ast.parse(f.read())

for node in ast.walk(tree):
    if isinstance(node, ast.List):
        if len(node.elts) == 15:
            print("Found a list of 15 elements at line", node.lineno)
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    print("  ", elt.id)
                elif isinstance(elt, ast.Call) and getattr(elt.func, 'id', None):
                    print("  ", elt.func.id)
                else:
                    print("  ", type(elt))
