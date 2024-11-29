from abc import ABC, abstractmethod

class FEPSimulation(ABC):
    """
    Abstract base class for Free Energy Perturbation (FEP) simulations.
    """
    _system = None
    _component_letter = None
    _available_methods = ['MBAR', 'TI-GQ', 'Analytical']

    def __init__(self,
                 sim_type: str,
                 temperature: float,
                 window: int,
                 weight: float,
                 path: str):
        """
        Initialize the FEP simulation with common parameters.
        
        Parameters
        ----------
        sim_type : str
            The type of simulation box to simulate. 
            It can be 'water' or 'membrane'.
        
        temperature : float
            The temperature of the simulation in Kelvin.

        window : int
            The window number of the simulation.

        weight : float
            The weight of the simulation.
        
        path : str
            The path to the simulation files.
        """
        self.sim_type = sim_type
        self.temperature = temperature
        self.window = window
        self.weight = weight
        self.path = path

    @classmethod
    @property
    def system(cls):
        """Class property of what is included in the simulation.
        It can be e.g. a complex system, an apo-protein, or a solvated lipid box.
        """
        return cls._system

    @classmethod
    @property
    def component_letter(self):
        """Class property of the component letter."""
        return self._component_letter
    
    @classmethod
    @property
    def available_methods(cls):
        """Class property of the available methods for the simulation."""
        return cls._available_methods

    @abstractmethod
    def run_simulation(self):
        """Abstract method to run the simulation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def setup(self):
        """Abstract method to set up the simulation. Must be implemented by subclasses."""
        pass

    def info(self):
        """Print the information of the simulation."""
        print(f"Simulation type: {self.sim_type}")
        print(f"Temperature: {self.temperature} K")
        print(f"Window: {self.window}")
        print(f"Weight: {self.weight}")

# Example of a subclass
class m_MFEPSimulation(FEPSimulation):
    """
    Merged FEP simulation of
    - attaching protein conformational restraints
    - attaching ligand conformational restraints
    - attaching ligand translational/rotational (TR) restraints
    """
    _system = 'complex'
    _component_letter = 'm'


class n_MFEPSimulation(FEPSimulation):
    """
    Merged FEP simulation of
    - releasing protein conformational restraints
    - releasing ligand conformational restraints
    """
    _system = 'complex'
    _component_letter = 'n'
    _available_methods = ['MBAR']


class e_MFEPSimulation(FEPSimulation):
    """
    Merged FEP simulation of
    - ligand charge decoupling in site
    - ligand charge recoupling in bulk
    """
    _system = 'complex'
    _component_letter = 'e'
    _available_methods = ['MBAR', 'TI-GQ']


class v_MFEPSimulation(FEPSimulation):
    """
    Merged FEP simulation of
    - ligand LJ decoupling in site
    - ligand LJ recoupling in bulk
    """
    _system = 'complex'
    _component_letter = 'v'
    _available_methods = ['MBAR', 'TI-GQ']


class a_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - attaching protein conformational restraints
    """
    _system = 'complex'
    _component_letter = 'a'
    _available_methods = ['MBAR']


class l_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - attaching ligand conformational restraints
    """
    _system = 'complex'
    _component_letter = 'l'
    _available_methods = ['MBAR']


class t_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - attaching ligand TR restraints
    """
    _system = 'complex'
    _component_letter = 't'
    _available_methods = ['MBAR']


class e_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - ligand charge decoupling in site
    """
    _system = 'complex'
    _component_letter = 'e'
    _available_methods = ['MBAR', 'TI-GQ']


class f_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - ligand charge recoupling in bulk
    """
    _system = 'ligand'
    _component_letter = 'f'
    _available_methods = ['MBAR', 'TI-GQ']


class v_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - ligand LJ decoupling in site
    """
    _system = 'complex'
    _component_letter = 'v'
    _available_methods = ['MBAR', 'TI-GQ']


class w_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - ligand LJ recoupling in bulk
    """
    _system = 'ligand'
    _component_letter = 'w'
    _available_methods = ['MBAR', 'TI-GQ']


class b_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - releasing protein conformational restraints
    """
    _system = 'ligand'
    _component_letter = 'b'
    _available_methods = ['Analytical']


class c_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - releasing ligand conformational restraints
    """
    _system = 'ligand'
    _component_letter = 'c'
    _available_methods = ['MBAR']


class r_FEPSimulation(FEPSimulation):
    """
    FEP simulation of
    - releasing protein conformational restraints
    """
    _system = 'protein'
    _component_letter = 'r'
    _available_methods = ['MBAR']