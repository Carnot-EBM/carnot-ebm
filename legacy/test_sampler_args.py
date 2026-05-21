from python.carnot.samplers.parallel_ising import ParallelIsingSampler
import inspect
print(inspect.signature(ParallelIsingSampler.sample))
print(inspect.signature(ParallelIsingSampler.__init__))
