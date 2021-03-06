import qmpy_rester as qr
import json
import time
import os

PAGE_LIMIT = 10

if not os.path.exists('query_files'):
    os.mkdir('query_files')

def download_by_batch(batch_num):
    t1 = time.time()
    with qr.QMPYRester() as q:
        kwargs = {'limit':PAGE_LIMIT, 
                  'offset': batch_num*PAGE_LIMIT,
                  'elements': 'He',
                  'fields':'_oqmd_entry_id,_oqmd_unit_cell,_oqmd_sites',
                  }
        data = q.get_optimade_structures(verbose=False, **kwargs)
    t2 = time.time()

    if batch_num == 0:
        print('Size of query dataset is %d.'%data['meta']['data_available'])
    
    with open('query_files/query_'+str(batch_num)+'.json', 'w') as json_file:
        json.dump(data['data'], json_file, indent=2)
    
    print('Loading Batch %d time %.3f seconds'%(batch_num, t2-t1))

    if data['links']['next']:
        return True
    else:
        return False

if __name__ == "__main__":
    batch_num = 0
    while download_by_batch(batch_num):
        batch_num = batch_num + 1
