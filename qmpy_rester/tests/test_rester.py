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
