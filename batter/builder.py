import batter
from batter.input_process import SimulationConfig, get_configure_from_file


class SystemBuilder:
    def __init__(self,
                 system: batter.System,
                 sim_config: SimulationConfig,
    ):
        self.system = system
        self.sim_config = sim_config
        