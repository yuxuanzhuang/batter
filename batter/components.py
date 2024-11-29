class Components:
    """
    One Component of a set of FEP simulations.
    It will contains a list of `FEPSimulations` with different
    windows and weights.
    """
    def __init__(self, component_letter: str):
        """
        Initialize the component with a letter.
        
        Parameters
        ----------
        component_letter : str
            The component letter.
        """
        self.component_letter = component_letter
        self.simulations = []

    def add_simulation(self, simulation: FEPSimulation):
        """
        Add a simulation to the component.
        
        Parameters
        ----------
        simulation : FEPSimulation
            The simulation to add.
        """
        self.simulations.append(simulation)