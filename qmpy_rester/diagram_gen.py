from .rester import *
from .phase_diagram import *

class PhaseDiagram(object):
    def __init__(self, space):
        self.space = space

    def get_phase_data(self):
        with QMPYRester() as q:
            ps_data = q.get_oqmd_phase_space(self.space)
    
        pd = PhaseData()
        pd.read_api_data(ps_data)

        self.phasedata = pd

    def add_phase(self, composition, energy, per_atom=True, **kwargs):
        p = Phase(composition=composition, energy=energy, per_atom=per_atom)
        self.phasedata.add_phase(p)

    @property
    def phases(self):
        return self.phasedata.phases

    @property
    def phase_dict(self):
        return self.phasedata.phase_dict

