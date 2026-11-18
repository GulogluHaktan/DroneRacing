"""Smoke test: imports the core package and the unit tests' dependencies."""
import torch, numpy
import droneracing
print("torch", torch.__version__, "| numpy", numpy.__version__, "| cuda", torch.cuda.is_available())
print("droneracing import OK")
