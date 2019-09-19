from unittest import TestCase

import qmpy_rester as qr

class TestRester(TestCase):
    def test_rester_oqmdapi_output_dict(self):
        with qr.QMPYRester() as q:
            kwargs = {'limit': '1'}
            data = q.get_oqmd_phases(verbose=False, **kwargs)
        self.assertTrue(isinstance(data, dict))

    def test_rester_oqmdapi_phase_space_output_dict(self):
        with qr.QMPYRester() as q:
            data = q.get_oqmd_phase_space('Pd-O')
        self.assertTrue(isinstance(data, dict))

    def test_rester_optimade_output_dict(self):
        with qr.QMPYRester() as q:
            kwargs = {'limit': '1'}
            data = q.get_optimade_structures(verbose=False, **kwargs)
        self.assertTrue(isinstance(data, dict))

    def test_rester_oqmdapi_by_id_output(self):
        with qr.QMPYRester() as q:
            data = q.get_oqmd_phase_by_id(fe_id=4061139,fields='name')
        self.assertEqual(data, {'name':'CsHoSiS4'})

    def test_rester_optimade_by_id_output(self):
        with qr.QMPYRester() as q:
            data = q.get_optimade_structure_by_id(id=4061139,fields='id,chemical_formula')
        self.assertEqual(data, {'id': 4061139, 'chemical_formula':'Cs1Ho1S4Si1'})

class TestPhaseDiagram(TestCase):
    def test_phase_diagram_creation(self):
        d = qr.PhaseDiagram('Pd-O')
        d.get_phase_data()

        self.assertTrue(isinstance(d.tie_lines, list))
        for t in d.tie_lines:
            self.assertTrue(isinstance(t[0], qr.Phase))
            self.assertTrue(isinstance(t[1], qr.Phase))

        self.assertTrue(isinstance(d.stable, set))
        for p in d.stable:
            self.assertTrue(isinstance(p, qr.Phase))

        self.assertTrue(isinstance(d.unstable, list))
        for p in d.unstable:
            self.assertTrue(isinstance(p, qr.Phase))
