import os
import sys


THIS_DIR = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
