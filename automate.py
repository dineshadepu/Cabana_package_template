#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import h5py

from itertools import cycle, product
import json
from automan.api import Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
# from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)
from automan.automation import (CommandTask)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'sans-serif', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


n_core = 6
# n_core = 16
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params

def get_files(directory):
    # =====================================
    # start: get the files and sort
    # =====================================
    files = [filename for filename in os.listdir(directory) if filename.startswith("particles") and filename.endswith("h5") ]
    files.sort()
    files_num = []
    for f in files:
        f_last = f[10:]
        files_num.append(int(f_last[:-3]))
    files_num.sort()

    sorted_files = []
    for num in files_num:
        sorted_files.append("particles_" + str(num) + ".h5")
    files = sorted_files
    return files


class Test01NestedForLoopPython(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'test01_nested_for_loop_python'

    def setup(self):
        get_path = self.input_path

        cmd = 'python python_src/nested_for_loop_and_for_loop.py $output_dir'
        # Base case info
        self.case_info = {
            'case_1': (dict(
                ), 'python'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()


class Test01NestedForLoopCabana(Problem):
    def get_name(self):
        return 'test01_nested_for_loop_cabana'

    def setup(self):
        get_path = self.input_path

        # cmd = './build/examples/TestComparePython01NestedLoop $output_dir'
        cmd = 'python examples/test_compare_python_01_nested_for_loop.py $output_dir'

        # Base case info
        self.case_info = {
            'case_1': (dict(
                ), 'Cabana'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_time_vs_normal_force()

    def plot_time_vs_normal_force(self):
        # get total no of particles
        cabana = h5py.File("outputs/test01_nested_for_loop_cabana/case_1/particles_0.h5", "r")
        python = np.load("outputs/test01_nested_for_loop_python/case_1/results.npz", "r")
        python_x = python['x']
        python_y = python['y']
        python_sum = python['total']

        cabana_x = cabana['positions'][:, 0]
        cabana_y = cabana['positions'][:, 1]
        cabana_radius = cabana['radius'][:]
        # print(cabana_radius)
        np.testing.assert_almost_equal(cabana_x, python_x)
        np.testing.assert_almost_equal(cabana_y, python_y)
        np.testing.assert_almost_equal(cabana_radius, python_sum)


if __name__ == '__main__':
    PROBLEMS = [
        # Image generator
        Test01NestedForLoopPython,
        Test01NestedForLoopCabana,
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
