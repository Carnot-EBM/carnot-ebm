import sys
sys.path.insert(0, 'python')
from carnot.verify.tier0e_eorm import EORMVerifier

def main():
    verifier = EORMVerifier()
    prob = verifier.verify("What is 2+3? **Answer:** 5")
    print(f"Prob: {prob}")

if __name__ == "__main__":
    main()
