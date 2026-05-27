prompts = build_prompts(questions)
for p in prompts:
    output = model(p)
