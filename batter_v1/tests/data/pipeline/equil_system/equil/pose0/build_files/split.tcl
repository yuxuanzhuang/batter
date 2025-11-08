mol load pdb rec_file.pdb
set dum [atomselect 0 "resname DUM"]
set prot [atomselect 0 "protein and not resname 'g1i'"]
set othrs [atomselect 0 "resname XXX and not water and same residue as within 15.00 of (protein or resname 'g1i')"]
set lipid [atomselect 0 "resname POPC PA PC OL"]
set wat [atomselect 0 "water and same residue as within 15.00 of (protein or resname 'g1i' or resname XXX or resname POPC PA PC OL)"]
set ion [atomselect 0 "resname 'Na+' 'Cl-' 'K+' and same residue as within 5 of (protein)"]
$wat set resname WAT
$dum writepdb dummy.pdb
$prot writepdb protein.pdb
$ion writepdb others.pdb
$lipid writepdb lipids.pdb
$wat writepdb crystalwat.pdb
exit
