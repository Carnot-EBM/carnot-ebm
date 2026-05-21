import sys

def append_to_table(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_row1 = "| CASAL Primal-Dual sampler implementation | **Tier 4 scaling validated** | Exp 1688/1690 |\n"
    new_row2 = "| EBFT continuous self-learning loop (Gemma 4) | **Baseline extended** | Exp 1692 |\n"
    new_row3 = "| SineKAN constraint splines | **Optimized verification pipeline** | Exp 1694 |\n"
    new_row4 = "| THRML/Carnot Curie-Weiss parity (n=128) | **Analytic ground truth met** | Exp 1692 |\n"
    new_row5 = "| Phase 1 Ship Readiness | **MCP Server + CLI Docs + HF Publication complete** | Exp 1695/1701 |\n"

    new_rows = new_row1 + new_row2 + new_row3 + new_row4 + new_row5

    # find the end of the table
    target = "| Tier 1 constraint addition v2 | Added 1 high-signal constraint; precision improved **0.478 → 0.917** and FPR dropped **0.857 → 0.071** on 50 held-out cases | Exp 1212 |\n"
    
    if target in content and "CASAL Primal-Dual" not in content:
        content = content.replace(target, target + new_rows)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Successfully appended to table in README.md")
    else:
        print("Target row not found or already appended.")

append_to_table('README.md')
