from importlib import resources

_base_ref = resources.files('batter.tests.data')

# Input files
example_input_files = (_base_ref / 'example_input')

## Correct input for test
ABFE = (example_input_files / 'abfe.in')
ABFE_UNO = (example_input_files / 'abfe_uno.in')
ABFE_UNOREST = (example_input_files / 'abfe_unorest.in')

ABFE_BUFFER0 = (example_input_files / 'abfe_buffer0.in')

## Incorrect input for test
INPUT_NUMWATER = (example_input_files / 'abfe_nwat.in')

# System files
B2AR_CAU_INPUT = (_base_ref / '2RH1')

# Ligand_files
LIGAND_FILES = (_base_ref / 'ligand_data')
KW6356 = (LIGAND_FILES / 'KW6356.sdf')
KW6356_H = (LIGAND_FILES / 'KW6356_h.sdf')