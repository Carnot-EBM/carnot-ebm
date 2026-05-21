import sys
sys.path.insert(0, 'python')
import carnot.verify
print("carnot.verify.__all__:", len(getattr(carnot.verify, '__all__', [])))
print("all items:", getattr(carnot.verify, '__all__', []))
