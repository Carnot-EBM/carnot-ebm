import os

def count_tests():
    count = 0
    for root, _, files in os.walk('tests/python'):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        if 'def test_' in line:
                            count += 1
    return count

print("Python test items:", count_tests())
