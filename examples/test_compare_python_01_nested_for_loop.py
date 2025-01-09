"""
"""
import shutil
import sys
import os
import numpy as np
import subprocess

output_dir = sys.argv[1]

shutil.copy('./build/examples/TestComparePython01NestedLoop', output_dir)

os.system('cd ' + output_dir + '&& ./TestComparePython01NestedLoop')
