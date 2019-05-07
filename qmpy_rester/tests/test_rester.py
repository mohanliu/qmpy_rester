from unittest import TestCase

import qmpy_rester as qr

class TestRester(TestCase):
    def test_rester_output_is_dict(self):
        with qr.QMPYRester() as q:
            kwargs = {'limit': '1'}
            data = q.get_oqmd_phases(verbose=False, **kwargs)
        self.assertTrue(isinstance(data, dict))
