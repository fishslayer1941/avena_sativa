import sys
import os

# Make the usda_pipeline/ directory importable from any pytest invocation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
