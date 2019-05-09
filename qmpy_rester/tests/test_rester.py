from unittest import TestCase

import qmpy_rester as qr

class TestRester(TestCase):
    def test_rester_oqmdapi_output_dict(self):
        with qr.QMPYRester() as q:
            kwargs = {'limit': '1'}
            data = q.get_oqmd_phases(verbose=False, **kwargs)
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
