import os
import numpy as np
from loguru import logger

class ComponentFEResult:
    _component_keys = ['fe', 'attach', 'elec', 'lj',
                       'release', 'boresch', 'uno', 'uno-rest']

    def __init__(self, fe_value, fe_std):
        """
        Class to store the results of the free energy calculations
        """
        self._results = {}
        # Example initialization; customize as needed
        for key in self._component_keys:
            self._results[key] = (fe_value, fe_std)

    @property
    def results(self):
        """
        Return the results
        """
        return self._results


    def __repr__(self):
        return f"FE: {self.fe if 'fe' in self._results else 'not calculated'} kcal/mol"
    
    def to_dict(self):
        json_dict = {}
        for key in self._component_keys:
            if key in self._results:
                json_dict[key] = {
                    'value': self._results[key][0],
                    'std': self._results[key][1]
                }
            else:
                json_dict[key] = {
                    'value': None,
                    'std': None
                }
        return json_dict
            

# Dynamically add properties
def _make_property(name, index):
    def getter(self):
        return self._results[name][index] if name in self._results else None
    return property(getter)

for key in ComponentFEResult._component_keys:
    setattr(ComponentFEResult, f"{key}", _make_property(key, 0))
    setattr(ComponentFEResult, f"{key}_std", _make_property(key, 1))


class FEResult(ComponentFEResult):
    """
    Class to store the results of all the free energy calculations
    that are generated with the old analysis.
    """
    def __init__(self, result_file):
        """
        Currently, the results are created from a file.
        """
        self.result_file = result_file
        if not os.path.exists(self.result_file):
            logger.error(f"File {self.result_file} does not exist")
            return
        self._results = {}

        self._read_results()

    def _read_results(self):
        """
        Read the results from the file
        The results will be stored as dictionary
        where the key is the component
        and the value is the free energy tuple (mean, std)
        """
        with open(self.result_file, 'r') as f:
            result_lines = f.readlines()
        if 'FAILED' in result_lines[0]:
            raise ValueError("Analysis failed")
        elif 'UNBOUND' in result_lines[0]:
            self._results = {}
            
            for key in self._component_keys:
                self._results[key] = ('unbound', 'unbound')
            return
        
        results = {}
        for line in result_lines:
            if line.startswith('Boresch'):
                comp = 'boresch'
            elif line.startswith('e'):
                comp = 'elec'
            elif line.startswith('v'):
                comp = 'lj'
            elif line.startswith('o'):
                comp = 'uno'
            elif line.startswith('n'):
                comp = 'release'
            elif line.startswith('m'):
                comp = 'attach'
            elif line.startswith('z'):
                comp = 'uno-rest'
            elif line.startswith('Total'):
                comp = 'fe'
            else:
                comp = None
            if comp is not None:
                energy = line.split()[-2][:-1]
                std = line.split()[-1]
                if energy == 'na':
                    energy = np.nan
                    std = np.nan
                    results[comp] = (energy, std)
                else:
                    results[comp] = (float(energy), float(std))
        self._results = results
