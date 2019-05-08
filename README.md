# qmpy_rester  [ ![Build Status](https://travis-ci.org/mohanliu/qmpy_rester.svg?branch=master) ](https://travis-ci.org/mohanliu/qmpy_rester)
A python code to query OQMD data through oqmd-api ([PyPI](https://pypi.org/project/qmpy-rester/)). This code is written in python 3.

## Installation
`pip install qmpy_rester`

## Usage
### 1. Get data through omqd-api
#### 1.1 Example 
```
import qmpy_rester as qr

with qr.QMPYRester() as q:
    kwargs = {
        ‘element_set’: ‘(Fe-Mn),O’,      # composition include (Fe OR Mn) AND O
        ‘stability’: ‘<-0.1’,            # hull distance smaller than -0.1 eV
        ‘natom’: ‘<10’,                  # number of atoms less than 10
        }
    list_of_data = q.get_oqmd_phases(**kwargs)
```
#### 1.2 Allowed attributes
- `composition`: compostion of the materials or phase space, e.g. Al2O3, Fe-O
- `element_set`: the set of elements that the compound must have, '-' for OR, ',' for AND, e.g. (Fe-Mn),O
- `icsd`: whether the structure exists in ICSD, e.g. False, True, F, T
- `prototype`: structure prototype of that compound, e.g. Cu, CsCl
- `generic`: chemical formula abstract, e.g. AB, AB2
- `spacegroup`: the space group of the structure, e.g. Fm-3m
- `natoms`: number of atoms in the supercell, e.g. 2, >5
- `volume`: volume of the supercell, e.g. >10
- `ntypes`: number of elements types in the compound, e.g. 2, <3
- `stability`: hull distance of the compound, e.g. 0, <-0.1,
- `delta_e`: formation energy of that compound, e.g. <-0.5,
- `band_gap`: band gap of the materials, e.g. 0, >2
- `filter`: customized filters, e.g. 'element_set=O AND ( stability<-0.1 OR delta_e<-0.5 )'
- `limit`: number of data return at once
- `offset`: the offset of data return

### 2. Get data through optimade api format
#### 2.1 Example 
```
import qmpy_rester as qr

with qr.QMPYRester() as q:
    kwargs = {
        ‘elements’: ‘Fe,Mn’,                    # include element Fe and Mn
        ‘nelements’: ‘<5’,                      # less than 4 element species in the compound
        ‘_oqmd_stability’: ‘<0’,                # stability calculted by oqmd is less than 0
        }
    list_of_data = q.get_optiamde_structures(**kwargs)
```
#### 1.2 Allowed attributes
- `elements`: the set of elements that the compound must have, e.g. Si,O
- `nelements`: number of elements types in the compound, e.g. 2, <3
- `chemical_formula`: compostion of the materials, e.g. Al2O3
- `formula_prototype`: chemical formula abstract, e.g. AB, AB2
- `_oqmd_natoms`: number of atoms in the supercell, e.g. 2, >5
- `_oqmd_volume`: volume of the supercell, e.g. >10
- `_oqmd_spacegroup`: the space group of the structure, e.g. Fm-3m
- `_oqmd_prototype`: structure prototype of that compound, e.g. Cu, CsCl
- `_oqmd_stability`: hull distance of the compound, e.g. 0, <-0.1,
- `_oqmd_delta_e`: formation energy of that compound, e.g. <-0.5,
- `_oqmd_band_gap`: band gap of the materials, e.g. 0, >2
- `filter`: customized filters, e.g. 'elements=O AND ( \_oqmd_stability<-0.1 OR \_oqmd_delta_e<-0.5 )'
- `limit`: number of data return at once
- `offset`: the offset of data return
