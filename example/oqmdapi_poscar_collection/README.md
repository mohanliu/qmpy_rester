## Usage

- Download data:
  - `python download_data.py`: 
    - This script will download data throught OQMDAPI, and save JSON files into 'query_files' folder.
    - Wisely choose `PAGE_LIMIT`. For large data query, `PAGE_LIMIT=2000` is recommended.

- Convert to POSCARs:
  - `python convert_to_poscars.py`
    - This script will convert the downloaded data into POSCAR files into 'poscars' folder.
    - Each POSCAR is named as 'POSCAR_'+_\_oqmd_entry\_id_

