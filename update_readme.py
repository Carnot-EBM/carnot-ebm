import re

with open("README.md", "r") as f:
    content = f.read()

if "## Phase 1 Milestone" not in content:
    match = re.search(r'(## .*?\n)(.*?\n\n)(## )', content, re.DOTALL)
    if match:
        new_content = content[:match.start(3)] + "## Phase 1 Milestone\n\nCarnot v0.1.0b1 marks Phase 1 completion: the carnot-ebm package on PyPI, HuggingFace mirror (huggingface.co/Carnot-EBM), ensemble verifier validation, MCP server, CLI, and Apache-2.0 license. The verifier pipeline runs on live GGUF outputs from state-of-the-art models (Qwen3.6-35B, Gemma-4-31B). See RELEASES.md for changelog.\n\n" + "## " + content[match.start(3)+3:]
        with open("README.md", "w") as f:
            f.write(new_content)
        print("added")
    else:
        print("no match")
else:
    print("already exists")
