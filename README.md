# qmpy_rester
A python code to query OQMD data through oqmd-api ([PyPI](https://pypi.org/project/qmpy-rester/)). This code is written in python 3.

## Installation
`pip install qmpy_rester`

## Usage
### Get data through omqd-api
```
import qmpy_rester as qr

with qr.QMPYRester() as q:
    kwargs = {
        ‘element_set’: ‘(Fe-Mn),O’,      # composition include (Fe OR Mn) AND O
        ‘stability’: ‘<-0.1’,            # hull distance smaller than -0.1 eV
        ‘natom’: ‘10’,                  # number of atoms less than 10
        }
    list_of_data = q.get_oqmd_phases(**kwargs)
```
