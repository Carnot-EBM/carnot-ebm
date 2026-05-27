import os
import sys

questions = load_questions()
for q in questions:
    result = infer(q)
