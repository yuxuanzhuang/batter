import os
import numpy as np
from loguru import logger

class ComponentFEResult:
    def __init__(self, fe_value, fe_std):
        """
        Class to store the results of the free energy calculations
        """
        self._results = {}
        self._results['fe'] = (fe_value, fe_std)
        self._results['attach'] = (fe_value, fe_std)
    
    @property
    def results(self):
        """
        Return the results
        """
        return self._results
    
    @property
    def fe(self):
        """
        Return the free energy tuple
        """
        return self._results['fe'][0] if 'fe' in self._results else None
    
    @property
    def fe_std(self):
        """
        Return the free energy standard deviation
        """
        return self._results['fe'][1] if 'fe' in self._results else None
    
    @property
    def attach_fe(self):
        """
        Return the attach energy tuple
        """
        return self._results['attach'][0] if 'attach' in self._results else None
    
    @property
    def attach_fe_std(self):
        """
        Return the attach energy standard deviation
        """
        return self._results['attach'][1] if 'attach' in self._results else None

    @property
    def elec_fe(self):
        """
        Return the electrostatic energy tuple
        """
        return self._results['elec'][0] if 'elec' in self._results else None

    @property
    def elec_fe_std(self):
        """
        Return the electrostatic energy standard deviation
        """
        return self._results['elec'][1] if 'elec' in self._results else None

    @property
    def lj_fe(self):
        """
        Return the Lennard-Jones energy tuple
        """
        return self._results['lj'][0] if 'lj' in self._results else None

    @property
    def lj_fe_std(self):
        """
        Return the Lennard-Jones energy standard deviation
        """
        return self._results['lj'][1] if 'lj' in self._results else None

    @property
    def release_fe(self):
        """
        Return the release energy tuple
        """
        return self._results['release'][0] if 'release' in self._results else None

    @property
    def release_fe_std(self):
        """
        Return the release energy standard deviation
        """
        return self._results['release'][1] if 'release' in self._results else None
    
    def __repr__(self):
        return f"FE: {self.fe if 'fe' in self._results else 'not calculated'} kcal/mol"


    
class FEResult(ComponentFEResult):
    """
    Class to store the results of all the free energy calculations
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
        if 'UNBOUND' in result_lines[0]:
            self._results = {}
            self._results['fe'] = (np.nan, np.nan)
            self._results['attach'] = (np.nan, np.nan)
            self._results['elec'] = (np.nan, np.nan)
            self._results['lj'] = (np.nan, np.nan)
            self._results['release'] = (np.nan, np.nan)
            return
        
        results = {}
        for line in result_lines:
            if line.startswith('Attach all'):
                comp = 'attach'
            elif line.startswith('Electrostatic'):
                comp = 'elec'
            elif line.startswith('Lennard-Jones'):
                comp = 'lj'
            elif line.startswith('LJ exchange'):
                comp = 'lj'
            elif line.startswith('Release all'):
                comp = 'release'
            elif line.startswith('Relative free energy'):
                comp = 'fe'
            elif line.startswith('Binding free energy'):
                comp = 'fe'
            else:
                comp = None
            if comp is not None:
                energy = line.split()[-2][:-1]
                std = line.split()[-1]
                results[comp] = (float(energy), float(std))
        self._results = results
