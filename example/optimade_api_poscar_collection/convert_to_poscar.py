import json
import os
import glob

if not os.path.exists('poscars'):
    os.mkdir('poscars')

def write_poscar(data, output):
    oqmd_id = data['_oqmd_entry_id']
    atoms = {}

    for s in data['_oqmd_sites']:
        ele = s.strip().split('@')[0].strip()
        pos = s.strip().split('@')[1].strip()

        if ele in atoms:
            atoms[ele].append(pos)
        else:
            atoms[ele] = [pos]

    with open(output, 'w') as f:
        f.write('oqmd_id_%d\n'%oqmd_id)
        f.write('1.0\n')

        for v in data['_oqmd_unit_cell']:
            f.write('%.3f %.3f %.3f\n' %tuple(v))

        atom_lst = atoms.keys()
        f.write('%s\n' %('  '.join(atom_lst)))
        f.write('%s\n' %('  '.join([str(len(atoms[v]))  \
                                    for v in atom_lst])))

        f.write('Direct\n')

        for a in atom_lst:
            f.write('%s\n' %('\n'.join(atoms[a])))

def poscar_conversion(query_file):
    with open(query_file, 'r') as j:
        data = json.load(j)
    
    for d in data:
        try:
            write_poscar(d, 'poscars/POSCAR_'+str(d['_oqmd_entry_id']))
        except:
            print('Failed to convert to POSCAR for entry %d'%(d['_oqmd_entry_id']))

if __name__ == "__main__":
    files = glob.glob('query_files/query_*.json')

    for f in files:
        poscar_conversion(f)
