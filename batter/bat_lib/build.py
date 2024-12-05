#!/usr/bin/env python2
import datetime as dt
import glob as glob
import os as os
import re as re
import shutil as shutil
import signal as signal
import subprocess as sp
import sys as sys
import batter.bat_lib.scripts as scripts
import filecmp
import MDAnalysis as mda
from batter.utils import (
    run_with_log,
    antechamber,
    tleap,
    cpptraj,
    parmchk2,
    charmmlipid2amber,
    obabel
)
from loguru import logger
import tempfile
import pandas as pd
import numpy as np
from batter.data import build_files as build_files_orig
from batter.data import amber_files as amber_files_orig
from batter.data import run_files as run_files_orig

def build_equil(pose, celp_st, mol,
                H1, H2, H3,
                calc_type,
                l1_x, l1_y, l1_z,
                l1_range, min_adis, max_adis,
                ligand_ff, ligand_ph,
                retain_lig_prot, ligand_charge,
                other_mol, solv_shell,
                lipid_mol, lipid_ff):

    # Not apply SDR distance when equilibrating
    sdr_dist = 0

    # Create equilibrium directory
    if not os.path.exists('equil'):
        os.makedirs('equil')
    os.chdir('equil')
    if os.path.exists('./build_files'):
        shutil.rmtree('./build_files')
    try:
        shutil.copytree(build_files_orig, './build_files')
    # Directories are the same
    except shutil.Error as e:
        logger.warning('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        logger.warning('Directory not copied. Error: %s' % e)
    os.chdir('build_files')

    if os.path.exists('../../all-poses/reference.pdb'):
        shutil.copy('../../all-poses/reference.pdb', './')
    else:
        # use itself as reference
        shutil.copy('../../all-poses/%s_docked.pdb' % (celp_st), './reference.pdb')

    if calc_type == 'dock':
        shutil.copy('../../all-poses/%s_docked.pdb' % (celp_st), './rec_file.pdb')
        shutil.copy('../../all-poses/%s.pdb' % (pose), './')
    elif calc_type == 'rank':
        shutil.copy('../../all-poses/%s.pdb' % (celp_st), './rec_file.pdb')
        shutil.copy('../../all-poses/%s.pdb' % (pose), './')
    elif calc_type == 'crystal':
        shutil.copy('../../all-poses/%s.pdb' % (pose), './')
        # Replace names and run initial VMD script
        with open("prep-crystal.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line.replace('MMM', mol).replace('mmm', mol.lower()).replace('CCCC', pose))
        run_with_log('vmd -dispdev text -e prep.tcl', error_match='anchor not found')

    # Split initial receptor file
    with open("split-ini.tcl", "rt") as fin:
        with open("split.tcl", "wt") as fout:
            if other_mol:
                other_mol_vmd = " ".join(other_mol)
            else:
                other_mol_vmd = 'XXX'
            if lipid_mol:
                lipid_mol_vmd = " ".join(lipid_mol)
            else:
                lipid_mol_vmd = 'XXX'
            for line in fin:
                if 'lig' not in line:
                    fout.write(line.replace('SHLL', '%4.2f' % solv_shell)
                        .replace('OTHRS', str(other_mol_vmd))
                        .replace('LIPIDS', str(lipid_mol_vmd))
                        .replace('MMM', mol.upper()))
    run_with_log('vmd -dispdev text -e split.tcl')

    # Remove possible remaining molecules
    if not other_mol:
        open('others.pdb', 'w').close()
    if not lipid_mol:
        open('lipids.pdb', 'w').close()

    shutil.copy('./protein.pdb', './protein_vmd.pdb')
    run_with_log('pdb4amber -i protein_vmd.pdb -o protein.pdb -y')
    renum_txt = 'protein_renum.txt'
    
    renum_data = pd.read_csv(
            renum_txt,
            sep='\s+',
            header=None,
            names=['old_resname', 'old_chain', 'old_resid',
                    'new_resname', 'new_resid'])

    # Get beginning and end of protein and save first residue as global variable
    u_vmd = mda.Universe('protein_vmd.pdb')
    first_res_vmd = int(u_vmd.residues[0].resid)

    # use the original receptor file to get the residue numbering
    # because terminal residues are removed during split.
    u_original = mda.Universe('rec_file.pdb')
    first_res = int(u_original.residues[0].resid)
    resid_list = u_original.residues.resids

    u_pdb4amber = mda.Universe('protein.pdb')
    recep_resid_num = len(u_pdb4amber.residues)

    logger.debug('Receptor first residue: %s' % first_res)
    logger.debug('Receptor total length: %s' % recep_resid_num)

    # Adjust protein anchors to the new residue numbering
    h1_resid = int(H1.split('@')[0][1:])
    h2_resid = int(H2.split('@')[0][1:])
    h3_resid = int(H3.split('@')[0][1:])

    h1_atom = H1.split('@')[1]
    h2_atom = H2.split('@')[1]
    h3_atom = H3.split('@')[1]

    protein_chain = 'A'
    h1_entry = (renum_data
            .query('old_resid == @h1_resid')
            .query('old_chain == @protein_chain')
    )
    h2_entry = (renum_data
            .query('old_resid == @h2_resid')
            .query('old_chain == @protein_chain')
    )
    h3_entry = (renum_data
            .query('old_resid == @h3_resid')
            .query('old_chain == @protein_chain')
    )
    if h1_entry.empty or h2_entry.empty or h3_entry.empty:
        logger.warning('Receptor is not set as chain A; '
                       'trying to fetch the first residue '
                       'from the protein and it may be wrong.')
        h1_entry = (renum_data
                .query('old_resid == @h1_resid')
        )
        h2_entry = (renum_data
                .query('old_resid == @h2_resid')
        )
        h3_entry = (renum_data
                .query('old_resid == @h3_resid')
        )
    # check not empty
    if h1_entry.empty or h2_entry.empty or h3_entry.empty:
        renum_data.to_csv('protein_renum_err.txt', sep='\t', index=False)  
        logger.error('Could not find the receptor anchors in the protein sequence')
        logger.error('Please check the residue numbering in the receptor file')
        logger.error(f'The renum is stored in {os.getcwd()}/protein_renum_err.txt')
        raise ValueError('Could not find the receptor anchors in the protein sequence')
    
    p1_resid = h1_entry['new_resid'].values[0]
    p2_resid = h2_entry['new_resid'].values[0]
    p3_resid = h3_entry['new_resid'].values[0]
    p1_vmd = f'{p1_resid}'

    P1 = f':{p1_resid}@{h1_atom}'
    P2 = f':{p2_resid}@{h2_atom}'
    P3 = f':{p3_resid}@{h3_atom}'

    logger.debug('Receptor anchors:')
    logger.info(f'P1: {P1}')
    logger.info(f'P2: {P2}')
    logger.info(f'P3: {P3}')

    # Replace names in initial files and VMD scripts
    # Here we convert all four letter residue names to three letter residue names
    if any(mol[:3] != mol for mol in other_mol):
        logger.warning(
            'The residue names of the co-binders are four-letter names.'
            'They were truncated to three-letter names'
            'for compatibility with AMBER.')
        other_mol = [mol[:3] for mol in other_mol]        

    if any(mol[:3] != mol for mol in lipid_mol):
        logger.warning(
            'The residue names of the lipids are four-letter names.'
            'They were truncated to three-letter names'
            'for compatibility with AMBER.')
        lipid_mol = [mol[:3] for mol in lipid_mol]

    # Save parameters in ff folder
    if not os.path.exists('../ff/'):
        os.makedirs('../ff/')
    for file in glob.glob('./*.mol2'):
        shutil.copy(file, '../ff/')
    for file in glob.glob('./*.frcmod'):
        shutil.copy(file, '../ff/')
    shutil.copy('./dum.mol2', '../ff/')
    shutil.copy('./dum.frcmod', '../ff/')

    # Adjust ligand files
    # Mudong's mod: optionally retain the ligand protonation state as provided in pose*.pdb, and skip Babel processing (removing H, adding H, determining total charge)
    if retain_lig_prot == 'yes':
        # Determine ligand net charge by reading the rightmost column of pose*.pdb, programs such as Maestro writes atom charges there
        run_with_log(f"{obabel} -i pdb {pose}.pdb -o mol2 -O {pose}.mol2")

        if ligand_charge == 'nd':
            ligand_mol2 = mda.Universe(f"{pose}.mol2")
            ligand_charge = np.round(np.sum(ligand_mol2.atoms.charges))
            
        logger.info('The net charge of the ligand is %d' % ligand_charge)
        if calc_type == 'dock' or calc_type == 'rank':
            shutil.copy(
            f'./{pose}.pdb',
            f'./{mol.lower()}-h.pdb')
        elif calc_type == 'crystal':
            shutil.copy(
            f'./{mol.lower()}.pdb',
            f'./{mol.lower()}-h.pdb')
        shutil.copy(
            f'./{pose}.mol2',
            f'./{mol.lower()}.mol2')

    else:
        if calc_type == 'dock' or calc_type == 'rank':
            run_with_log(f'{obabel} -i pdb {pose}.pdb -o pdb -O {mol.lower()}.pdb -d',
                         error_match='cannot read input format!')                            # Remove all hydrogens from the ligand
        elif calc_type == 'crystal':
            run_with_log(f'{obabel} -i pdb {mol.lower()}.pdb -o pdb -O {mol.lower()}.pdb -d',
                         error_match='cannot read input format!')                     # Remove all hydrogens from crystal ligand
        run_with_log(f'{obabel} -i pdb {mol.lower()}.pdb -o pdb -O {mol.lower()}-h-ini.pdb -p {ligand_ph:.2f}',
                     error_match='cannot read input format!')  # Put all hydrogens back using babel
        run_with_log(f'{obabel} -i pdb {mol.lower()}.pdb -o mol2 -O {mol.lower()}-crg.mol2 -p {ligand_ph:.2f}',
                     error_match='cannot read input format!')
        # Clean ligand protonated pdb file
        with open(mol.lower()+'-h-ini.pdb') as oldfile, open(mol.lower()+'-h.pdb', 'w') as newfile:
            for line in oldfile:
                if 'ATOM' in line or 'HETATM' in line:
                    newfile.write(line)
            newfile.close()
        if ligand_charge == 'nd':
            ligand_charge = 0
            # Get ligand net charge from babel
            lig_crg = 0
            with open('%s-crg.mol2' % mol.lower()) as f_in:
                for line in f_in:
                    splitdata = line.split()
                    if len(splitdata) > 8:
                        lig_crg = lig_crg + float(splitdata[8].strip())
            ligand_charge = round(lig_crg)
        
        shutil.copy(f'{mol.lower()}-crd.mol2', f'{mol.lower()}.mol2')
        logger.info('The babel protonation of the ligand is for pH %4.2f' % ligand_ph)
        logger.info('The net charge of the ligand is %d' % ligand_charge)

    # Get ligand parameters
    if not os.path.exists('../ff/%s.mol2' % mol.lower()):
 #        run_with_log(antechamber + ' -i '+mol.lower()+'-h.pdb -fi pdb -o '+mol.lower() +
 #                     '.mol2 -fo mol2 -c bcc -s 2 -at '+ligand_ff.lower()+' -nc %d' % ligand_charge)
        run_with_log(
            f'{antechamber} -i {mol.lower()}.mol2 -fi mol2 -o {mol.lower()}.mol2 '
            f'-fo mol2 -c bcc -s 2 -at {ligand_ff.lower()} -nc {ligand_charge}')
        shutil.copy(f'./{mol.lower()}.mol2', '../ff/')
    if not os.path.exists('../ff/%s.frcmod' % mol.lower()):
        if ligand_ff == 'gaff':
            run_with_log(parmchk2 + ' -i '+mol.lower()+'.mol2 -f mol2 -o '+mol.lower()+'.frcmod -s 1')
        elif ligand_ff == 'gaff2':
            run_with_log(parmchk2 + ' -i '+mol.lower()+'.mol2 -f mol2 -o '+mol.lower()+'.frcmod -s 2')
        shutil.copy('./%s.frcmod' % (mol.lower()), '../ff/')
        
 #    run_with_log(antechamber + ' -i '+mol.lower()+'-h.pdb -fi pdb -o '+mol.lower()+'.pdb -fo pdb')
    run_with_log(
        f'{antechamber} -i {mol.lower()}.mol2 -fi mol2 -o {mol.lower()}.pdb -fo pdb')

    # Convert CHARMM lipid into lipid21
    run_with_log(f'{charmmlipid2amber} -i lipids.pdb -o lipids_amber.pdb')
    u = mda.Universe('lipids_amber.pdb')
    lipid_resnames = set([resname for resname in u.residues.resnames])
    old_lipid_mol = list(lipid_mol)
    lipid_mol = list(lipid_resnames)
    logger.debug(f'Converting CHARMM lipids: {old_lipid_mol} to AMBER format: {lipid_mol}')

    with open("prep-ini.tcl", "rt") as fin:
        with open("prep.tcl", "wt") as fout:
            other_mol_vmd = " ".join(other_mol)
            lipid_mol_vmd = " ".join(lipid_mol)
            for line in fin:
                fout.write(line.replace('MMM', mol).replace('mmm', mol.lower())
                           .replace('NN', h1_atom)
                           .replace('P1A', p1_vmd)
                           .replace('FIRST', '1')
                           .replace('LAST', str(recep_resid_num))
                           .replace('STAGE', 'equil')
                           .replace('XDIS', '%4.2f' % l1_x)
                           .replace('YDIS', '%4.2f' % l1_y)
                           .replace('ZDIS', '%4.2f' % l1_z)
                           .replace('RANG', '%4.2f' % l1_range)
                           .replace('DMAX', '%4.2f' % max_adis)
                           .replace('DMIN', '%4.2f' % min_adis)
                           .replace('SDRD', '%4.2f' % sdr_dist)
                           .replace('OTHRS', str(other_mol_vmd))
                           .replace('LIPIDS', str(lipid_mol_vmd))
                           )

    # Create raw complex and clean it
    filenames = ['protein.pdb',
                 '%s.pdb' % mol.lower(),
                 'others.pdb',
                 'lipids_amber.pdb',
                 'crystalwat.pdb']
    with open('./complex-merge.pdb', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    with open('complex-merge.pdb') as oldfile, open('complex.pdb', 'w') as newfile:
        for line in oldfile:
            if not 'CRYST1' in line and not 'CONECT' in line and not 'END' in line:
                newfile.write(line)

    # New work around to avoid chain swapping during alignment
    run_with_log('pdb4amber -i reference.pdb -o reference_amber.pdb -y')
    run_with_log('vmd -dispdev text -e nochain.tcl')
    run_with_log('./USalign complex-nc.pdb reference_amber-nc.pdb -mm 0 -ter 2 -o aligned-nc')
    run_with_log('vmd -dispdev text -e measure-fit.tcl')

    # Put in AMBER format and find ligand anchor atoms
    with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
        for line in oldfile:
            splitdata = line.split()
            if len(splitdata) > 4:
                newfile.write(line)
    
    # Note: this command will convert four-letter residue names to three-letter residue names
    # This will also turn lipid partial group (PC, PA, OL)
    # that belong to the same residue number
    # into different residue number 
    run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

    # add box info back if lipid is present
    # also renumber the residue number
    # so that PA, PC, OL (for POPC) are in the same residue number
    if lipid_mol:
        u = mda.Universe('aligned_amber.pdb')
        box_origin = u_original.dimensions
        u.dimensions = box_origin
        logger.debug(f'Adding box info back to aligned_amber.pdb: {box_origin}')

        renum_txt = 'aligned_amber_renum.txt'
        
        renum_data = pd.read_csv(
                renum_txt,
                sep='\s+',
                header=None,
                names=['old_resname', 'old_chain', 'old_resid',
                       'new_resname', 'new_resid'])

        revised_resids = []
        resid_counter = 1
        prev_resid = 0
        for i, row in renum_data.iterrows():
            if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                revised_resids.append(resid_counter)
                resid_counter += 1
            else:
                revised_resids.append(resid_counter - 1)
            prev_resid = row['old_resid']
        u.atoms.residues.resids = revised_resids

        u.atoms.write('aligned_amber.pdb')

    run_with_log('vmd -dispdev text -e prep.tcl', error_match='anchor not found')

    # Check size of anchor file
    anchor_file = 'anchors.txt'
    if os.stat(anchor_file).st_size == 0:
        os.chdir('../')
        return 'anch1'
    f = open(anchor_file, 'r')
    for line in f:
        splitdata = line.split()
        if len(splitdata) < 3:
            os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
            os.chdir('../')
            return 'anch2'
    os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
    os.chdir('../')

    # Create simulation directory
    if not os.path.exists(pose):
        os.makedirs(pose)
    os.chdir(pose)

    dum_coords = []
    recep_coords = []
    lig_coords = []
    oth_coords = []
    dum_atomlist = []
    lig_atomlist = []
    recep_atomlist = []
    oth_atomlist = []
    dum_rsnmlist = []
    recep_rsnmlist = []
    lig_rsnmlist = []
    oth_rsnmlist = []
    dum_rsidlist = []
    recep_rsidlist = []
    lig_rsidlist = []
    oth_rsidlist = []
    dum_chainlist = []
    recep_chainlist = []
    lig_chainlist = []
    oth_chainlist = []
    dum_atom = 0
    lig_atom = 0
    recep_atom = 0
    oth_atom = 0
    total_atom = 0
    resid_lig = 0
    resname_lig = mol

    # Copy a few files
    shutil.copy('../build_files/equil-%s.pdb' % mol.lower(), './')

    # Use equil-reference.pdb to retrieve the box size
    shutil.copy('../build_files/equil-%s.pdb' % mol.lower(), './equil-reference.pdb')
    shutil.copy('../build_files/%s-noh.pdb' % mol.lower(), './%s.pdb' % mol.lower())
    shutil.copy('../build_files/anchors-'+pose+'.txt', './anchors.txt')

    # Read coordinates for dummy atoms
    for i in range(1, 2):
        shutil.copy('../build_files/dum'+str(i)+'.pdb', './')
        with open('dum'+str(i)+'.pdb') as dum_in:
            lines = (line.rstrip() for line in dum_in)
            lines = list(line for line in lines if line)
            dum_coords.append((float(lines[1][30:38].strip()), float(
                lines[1][38:46].strip()), float(lines[1][46:54].strip())))
            dum_atomlist.append(lines[1][12:16].strip())
            dum_rsnmlist.append(lines[1][17:20].strip())
            dum_rsidlist.append(float(lines[1][22:26].strip()))
            dum_chainlist.append(lines[1][21].strip())
            dum_atom += 1
            total_atom += 1

    # Read coordinates from aligned system
    with open('equil-%s.pdb' % mol.lower()) as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list

    # Count atoms of receptor and ligand
    for i in range(0, len(lines)):
        if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
            molecule = lines[i][17:21].strip()
            if molecule not in {mol, 'DUM', 'WAT'} and molecule not in other_mol and molecule not in lipid_mol:
                recep_coords.append((
                    float(lines[i][30:38].strip()),
                    float(lines[i][38:46].strip()),
                    float(lines[i][46:54].strip())))
                recep_atomlist.append(lines[i][12:16].strip())
                recep_rsnmlist.append(molecule)
                recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                recep_chainlist.append(lines[i][21].strip())
                recep_last = int(lines[i][22:26].strip())
                recep_atom += 1
                total_atom += 1
            elif molecule == mol:
                lig_coords.append((float(lines[i][30:38].strip()), float(
                    lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                lig_atomlist.append(lines[i][12:16].strip())
                lig_rsnmlist.append(molecule)
                lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                lig_chainlist.append(lines[i][21].strip())
                lig_atom += 1
                total_atom += 1
            elif (molecule == 'WAT') or (molecule in other_mol):
                oth_coords.append((float(lines[i][30:38].strip()), float(
                    lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                oth_atomlist.append(lines[i][12:16].strip())
                oth_rsnmlist.append(molecule)
                oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                oth_chainlist.append(lines[i][21].strip())
                oth_atom += 1
                total_atom += 1
            elif molecule in lipid_mol:
                oth_coords.append((float(lines[i][30:38].strip()), float(
                    lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                oth_atomlist.append(lines[i][12:16].strip())
                oth_rsnmlist.append(molecule)
                oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom)
                oth_chainlist.append(lines[i][21].strip())
                oth_atom += 1
                total_atom += 1

    coords = dum_coords + recep_coords + lig_coords + oth_coords
    atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
    resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
    resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
    chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
    lig_resid = str(recep_last + dum_atom + 1)
    chain_tmp = 'None'
    resid_tmp = 'None'

    # Read ligand anchors obtained from VMD
    anchor_file = 'anchors.txt'
    f = open(anchor_file, 'r')
    for line in f:
        splitdata = line.split()
        L1 = ":"+lig_resid+"@"+splitdata[0]
        L2 = ":"+lig_resid+"@"+splitdata[1]
        L3 = ":"+lig_resid+"@"+splitdata[2]

    logger.info('Ligand anchors:')
    logger.info(f'L1: {L1}')
    logger.info(f'L2: {L2}')
    logger.info(f'L3: {L3}')

    # Write the new pdb file
    build_file = open('build.pdb', 'w')

    # Positions for the dummy atoms
    for i in range(0, 1):
        build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                         ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
        build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
        build_file.write('%6.2f%6.2f\n' % (0, 0))
        build_file.write('TER\n')

    # Positions of the receptor atoms
    for i in range(dum_atom, dum_atom + recep_atom):
        if chain_list[i] != chain_tmp:
            if resname_list[i] not in other_mol and resname_list[i] != 'WAT':
                build_file.write('TER\n')
        chain_tmp = chain_list[i]
        build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                 ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
        build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

        build_file.write('%6.2f%6.2f\n' % (0, 0))
    build_file.write('TER\n')

    # Positions of the ligand atoms
    for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):
        build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                         ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
        build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

        build_file.write('%6.2f%6.2f\n' % (0, 0))

    build_file.write('TER\n')

    # Positions of the other atoms
    for i in range(dum_atom + recep_atom + lig_atom, total_atom):
        if resid_list[i] != resid_tmp:
            build_file.write('TER\n')
        resid_tmp = resid_list[i]
        build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                         ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
        build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

        build_file.write('%6.2f%6.2f\n' % (0, 0))

    build_file.write('TER\n')
    build_file.write('END\n')

    # Write anchors and last protein residue to original pdb file
    with open('equil-%s.pdb' % mol.lower(), 'r') as fin:
        data = fin.read().splitlines(True)
    with open('equil-%s.pdb' % mol.lower(), 'w') as fout:
        fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                   ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
        fout.writelines(data[1:])

    # Check for missing residues in receptor structure
    if recep_last != recep_resid_num:
        logger.warning('WARNING: Missing residues in the receptor protein sequence. Unless the protein is engineered this is not recommended,')
        logger.warning('a protein modeling tool might be required before running equilibration.')

    f_in.close()
    build_file.close()

    # Write dry build file

    with open('build.pdb') as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = list(line for line in lines if line)  # Non-blank lines in a list
    with open('./build-dry.pdb', 'w') as outfile:
        for i in range(0, len(lines)):
            if lines[i][17:20].strip() == 'WAT':
                break
            outfile.write(lines[i]+'\n')

    outfile.close()

    os.chdir('../')

    return 'all'


# Build the system for the production run
def build_dec(fwin, hmr, mol,
              pose, molr, poser,
              comp, win, water_model,
              ntpr, ntwr, ntwe, ntwx,
              cut, gamma_ln, barostat,
              receptor_ff, ligand_ff, dt,
              sdr_dist, dec_method, l1_x, l1_y, l1_z,
              l1_range, min_adis, max_adis,
              ion_def, other_mol, solv_shell,
              lipid_mol, lipid_ff):


    if lipid_mol:
        build_file_path = 'build_files'
        amber_files_path = './amber_files'
        p_coupling = '3'
        c_surften = '3'
    else:
        build_file_path = 'build_files_no_lipid'
        amber_files_path = './amber_files_no_lipid'
        p_coupling = '1'
        c_surften = '0'
    if comp == 'n':
        dec_method == 'sdr'

    if comp == 'a' or comp == 'l' or comp == 't' or comp == 'm' or comp == 'c' or comp == 'r':
        dec_method = 'dd'

    if comp == 'x':
        dec_method = 'exchange'

    # Get files or finding new anchors and building some systems
    if (not os.path.exists(f'../{build_file_path}')) or (dec_method == 'sdr' and win == 0) or (dec_method == 'exchange' and win == 0):
        if (dec_method == 'sdr' or dec_method == 'exchange') and os.path.exists('../build_files'):
            shutil.rmtree(f'../{build_file_path}')
        try:
            shutil.copytree('../../../equil/build_files',
                            f'../{build_file_path}')
        # Directories are the same
        except shutil.Error as e:
            logger.warning('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            logger.warning('Directory not copied. Error: %s' % e)
        os.chdir(f'../{build_file_path}')
        # Get last state from equilibrium simulations
        shutil.copy('../../../equil/'+pose+'/md%02d.rst7' % fwin, './')
        shutil.copy('../../../equil/'+pose+'/full.pdb', './aligned-nc.pdb')
        shutil.copy('../../../equil/'+pose+'/build_amber_renum.txt', './')
        for file in glob.glob('../../../equil/%s/full*.prmtop' % pose.lower()):
            shutil.copy(file, './')
        for file in glob.glob('../../../equil/%s/vac*' % pose.lower()):
            shutil.copy(file, './')
        run_with_log(cpptraj + ' -p full.prmtop -y md%02d.rst7 -x rec_file.pdb' % fwin)
        renum_data = pd.read_csv('build_amber_renum.txt', sep='\s+',
                header=None, names=['old_resname',
                                    'old_chain',
                                    'old_resid',
                                    'new_resname', 'new_resid'])
        u = mda.Universe('rec_file.pdb')

        for residue in u.select_atoms('protein').residues:
            resid_str = residue.resid
            residue.atoms.chainIDs = renum_data.query(f'old_resid == @resid_str').old_chain.values[0]

        if lipid_mol:

            # fix lipid resids
            revised_resids = []
            resid_counter = 1
            prev_resid = 0
            for i, row in renum_data.iterrows():
                if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                    revised_resids.append(resid_counter)
                    resid_counter += 1
                else:
                    revised_resids.append(resid_counter - 1)
                prev_resid = row['old_resid']
            
            renum_data['revised_resid'] = revised_resids
            revised_resids = np.array(revised_resids)
            total_residues = u.atoms.residues.n_residues
            final_resids = np.zeros(total_residues, dtype=int)
            final_resids[:len(revised_resids)] = revised_resids
            next_resnum = revised_resids[-1] + 1
            final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
            u.atoms.residues.resids = final_resids

        u.atoms.write('rec_file.pdb')

        # Used for retrieving the box size
        shutil.copy('rec_file.pdb', 'equil-reference.pdb')

        # convert back to lipid

        # Split initial receptor file
        with open("split-ini.tcl", "rt") as fin:
            with open("split.tcl", "wt") as fout:
                if other_mol:
                    other_mol_vmd = " ".join(other_mol)
                else:
                    other_mol_vmd = 'XXX'
                if lipid_mol:
                    lipid_mol_vmd = " ".join(lipid_mol)
                else:
                    lipid_mol_vmd = 'XXX'
                for line in fin:
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', mol.lower())
                    .replace('MMM', mol.upper()))
        run_with_log('vmd -dispdev text -e split.tcl')

        # Remove possible remaining molecules
        if not other_mol:
            open('others.pdb', 'w').close()
        if not lipid_mol:
            open('lipids.pdb', 'w').close()

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                     'protein.pdb',
                     '%s.pdb' % mol.lower(),
                     'others.pdb',
                     'lipids.pdb',
                     'crystalwat.pdb']
        with open('./complex-merge.pdb', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        with open('complex-merge.pdb') as oldfile, open('complex.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'CRYST1' in line and not 'CONECT' in line and not 'END' in line:
                    newfile.write(line)

        # Read protein anchors and size from equilibrium
        with open('../../../equil/'+pose+'/equil-%s.pdb' % mol.lower(), 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line
                        .replace('MMM', mol)
                        .replace('mmm', mol.lower())
                        .replace('NN', p1_atom)
                        .replace('P1A', p1_vmd)
                        .replace('FIRST', '2')
                        .replace('LAST', str(rec_res))
                        .replace('STAGE', 'fe')
                        .replace('XDIS', '%4.2f' % l1_x)
                        .replace('YDIS', '%4.2f' % l1_y)
                        .replace('ZDIS', '%4.2f' % l1_z)
                        .replace('RANG', '%4.2f' % l1_range)
                        .replace('DMAX', '%4.2f' % max_adis)
                        .replace('DMIN', '%4.2f' % min_adis)
                        .replace('SDRD', '%4.2f' % sdr_dist)
                        .replace('OTHRS', str(other_mol_vmd))
                        .replace('LIPIDS', str(lipid_mol_vmd))
                        )

        # Align to reference (equilibrium) structure using VMD's measure fit
        run_with_log('vmd -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # fix lipid resids
        if lipid_mol:
            u = mda.Universe('aligned_amber.pdb')
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep='\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
                        'new_resname', 'new_resid'])

            revised_resids = []
            resid_counter = 1
            prev_resid = 0
            for i, row in renum_data.iterrows():
                if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                    revised_resids.append(resid_counter)
                    resid_counter += 1
                else:
                    revised_resids.append(resid_counter - 1)
                prev_resid = row['old_resid']
            # set correct residue number
            revised_resids = np.array(revised_resids)
            u.atoms.residues.resids = final_resids[:len(revised_resids)]
            u.atoms.write('aligned_amber.pdb')
        
        run_with_log('vmd -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            os.chdir('../')
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+pose+'.txt')
                os.chdir('../')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+pose+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+pose+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % mol.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % mol.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])

        # Get parameters from equilibrium
        if not os.path.exists('../ff'):
            os.makedirs('../ff')
        for file in glob.glob('../../../equil/ff/*.mol2'):
            shutil.copy(file, '../ff/')
        for file in glob.glob('../../../equil/ff/*.frcmod'):
            shutil.copy(file, '../ff/')
        shutil.copy('../../../equil/ff/%s.mol2' % (mol.lower()), '../ff/')
        shutil.copy('../../../equil/ff/%s.frcmod' % (mol.lower()), '../ff/')
        shutil.copy('../../../equil/ff/dum.mol2', '../ff/')
        shutil.copy('../../../equil/ff/dum.frcmod', '../ff/')

        if (comp == 'v' or comp == 'e' or comp == 'w' or comp == 'f'):
            if dec_method == 'dd':
                os.chdir('../dd/')
            if dec_method == 'sdr' or dec_method == 'exchange':
                os.chdir('../sdr/')
        elif comp != 'x':
            os.chdir('../rest/')
        
    # Create reference for relative calculations
    if comp == 'x' and win == 0:

        # Build reference ligand from last state of equilibrium simulations

        if not os.path.exists('../exchange_files'):
            shutil.copytree(f'../../../{build_file_path}', '../exchange_files')
        os.chdir('../exchange_files')
        shutil.copy('../../../equil/'+poser+'/md%02d.rst7' % fwin, './')
        shutil.copy('../../../equil/'+pose+'/full.pdb', './aligned-nc.pdb')
        for file in glob.glob('../../../equil/%s/full*.prmtop' % poser.lower()):
            shutil.copy(file, './')
        for file in glob.glob('../../../equil/%s/vac*' % poser.lower()):
            shutil.copy(file, './')
        run_with_log(cpptraj + ' -p full.prmtop -y md%02d.rst7 -x rec_file.pdb' % fwin)

        # restore resid index
        
        shutil.copy('rec_file.pdb', 'equil-reference.pdb')

        # Split initial receptor file
        with open("split-ini.tcl", "rt") as fin:
            with open("split.tcl", "wt") as fout:
                if other_mol:
                    other_mol_vmd = " ".join(other_mol)
                else:
                    other_mol_vmd = 'XXX'
                if lipid_mol:
                    lipid_mol_vmd = " ".join(lipid_mol)
                else:
                    lipid_mol_vmd = 'XXX'
                for line in fin:
                    fout.write(line
                    .replace('SHLL', '%4.2f' % solv_shell)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    .replace('mmm', molr.lower())
                    .replace('MMM', molr.upper()))
        run_with_log('vmd -dispdev text -e split.tcl')

        # Remove possible remaining molecules
        if not other_mol:
            open('others.pdb', 'w').close()
        if not lipid_mol:
            open('lipids.pdb', 'w').close()

        # Create raw complex and clean it
        filenames = ['dummy.pdb',
                    'protein.pdb',
                    '%s.pdb' % molr.lower(),
                    'others.pdb',
                    'lipids.pdb',
                    'crystalwat.pdb']
        with open('./complex-merge.pdb', 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
        with open('complex-merge.pdb') as oldfile, open('complex.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'CRYST1' in line and not 'CONECT' in line and not 'END' in line:
                    newfile.write(line)

        # Read protein anchors and size from equilibrium
        with open('../../../equil/'+poser+'/equil-%s.pdb' % molr.lower(), 'r') as f:
            data = f.readline().split()
            P1 = data[2].strip()
            P2 = data[3].strip()
            P3 = data[4].strip()
            first_res = data[8].strip()
            recep_last = data[9].strip()

        # Get protein first anchor residue number and protein last residue number from equil simulations
        p1_resid = P1.split('@')[0][1:]
        p1_atom = P1.split('@')[1]
        rec_res = int(recep_last)+1
        p1_vmd = p1_resid

        # Replace names in initial files and VMD scripts
        with open("prep-ini.tcl", "rt") as fin:
            with open("prep.tcl", "wt") as fout:
                for line in fin:
                    fout.write(line
                    .replace('MMM', molr)
                    .replace('mmm', molr.lower())
                    .replace('NN', p1_atom)
                    .replace('P1A', p1_vmd)
                    .replace('FIRST', '2')
                    .replace('LAST', str(rec_res))
                    .replace('STAGE', 'fe')
                    .replace('XDIS', '%4.2f' % l1_x)
                    .replace('YDIS', '%4.2f' % l1_y)
                    .replace('ZDIS', '%4.2f' % l1_z)
                    .replace('RANG', '%4.2f' % l1_range)
                    .replace('DMAX', '%4.2f' % max_adis)
                    .replace('DMIN', '%4.2f' % min_adis)
                    .replace('SDRD', '%4.2f' % sdr_dist)
                    .replace('OTHRS', str(other_mol_vmd))
                    .replace('LIPIDS', str(lipid_mol_vmd))
                    )

        # Align to reference (equilibrium) structure using VMD's measure fit
        run_with_log('vmd -dispdev text -e measure-fit.tcl')

        # Put in AMBER format and find ligand anchor atoms
        with open('aligned.pdb', 'r') as oldfile, open('aligned-clean.pdb', 'w') as newfile:
            for line in oldfile:
                splitdata = line.split()
                if len(splitdata) > 3:
                    newfile.write(line)
        run_with_log('pdb4amber -i aligned-clean.pdb -o aligned_amber.pdb -y')

        # Fix lipid
        if lipid_mol:
            renum_txt = 'aligned_amber_renum.txt'
            
            renum_data = pd.read_csv(
                    renum_txt,
                    sep='\s+',
                    header=None,
                    names=['old_resname', 'old_resid',
                        'new_resname', 'new_resid'])

            revised_resids = []
            resid_counter = 1
            prev_resid = 0
            for i, row in renum_data.iterrows():
                if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
                    revised_resids.append(resid_counter)
                    resid_counter += 1
                else:
                    revised_resids.append(resid_counter - 1)
                prev_resid = row['old_resid']
            u.atoms.residues.resids = revised_resids

            u.atoms.write('aligned_amber.pdb')

        run_with_log('vmd -dispdev text -e prep.tcl', error_match='anchor not found')

        # Check size of anchor file
        anchor_file = 'anchors.txt'
        if os.stat(anchor_file).st_size == 0:
            os.chdir('../')
            return 'anch1'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            if len(splitdata) < 3:
                os.rename('./anchors.txt', 'anchors-'+poser+'.txt')
                os.chdir('../')
                return 'anch2'
        os.rename('./anchors.txt', 'anchors-'+poser+'.txt')

        # Read ligand anchors obtained from VMD
        lig_resid = str(int(recep_last) + 2)
        anchor_file = 'anchors-'+poser+'.txt'
        f = open(anchor_file, 'r')
        for line in f:
            splitdata = line.split()
            L1 = ":"+lig_resid+"@"+splitdata[0]
            L2 = ":"+lig_resid+"@"+splitdata[1]
            L3 = ":"+lig_resid+"@"+splitdata[2]

        # Write anchors and last protein residue to original pdb file
        with open('fe-%s.pdb' % molr.lower(), 'r') as fin:
            data = fin.read().splitlines(True)
        with open('fe-%s.pdb' % molr.lower(), 'w') as fout:
            fout.write('%-8s  %6s  %6s  %6s  %6s  %6s  %6s  %6s  %4s\n' %
                       ('REMARK A', P1, P2, P3, L1, L2, L3, first_res, recep_last))
            fout.writelines(data[1:])

        # Get parameters from equilibrium
        if not os.path.exists('../ff'):
            os.makedirs('../ff')
        for file in glob.glob('../../../equil/ff/*.mol2'):
            shutil.copy(file, '../ff/')
        for file in glob.glob('../../../equil/ff/*.frcmod'):
            shutil.copy(file, '../ff/')
        shutil.copy('../../../equil/ff/%s.mol2' % (molr.lower()), '../ff/')
        shutil.copy('../../../equil/ff/%s.frcmod' % (molr.lower()), '../ff/')
        shutil.copy('../../../equil/ff/dum.mol2', '../ff/')
        shutil.copy('../../../equil/ff/dum.frcmod', '../ff/')

        os.chdir('../sdr/')

    # Copy and replace simulation files for the first window
    if int(win) == 0:
        if os.path.exists(amber_files_path):
            shutil.rmtree(amber_files_path)
        try:
            shutil.copytree(amber_files_orig,
                            amber_files_path)
        # Directories are the same
        except shutil.Error as e:
            logger.warning('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            logger.warning('Directory not copied. Error: %s' % e)
        
        for dname, dirs, files in os.walk(amber_files_path):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (s.replace('_step_', dt)
                        .replace('_ntpr_', ntpr)
                        .replace('_ntwr_', ntwr)
                        .replace('_ntwe_', ntwe)
                        .replace('_ntwx_', ntwx)
                        .replace('_cutoff_', cut)
                        .replace('_gamma_ln_', gamma_ln)
                        .replace('_barostat_', barostat)
                        .replace('_receptor_ff_', receptor_ff)
                        .replace('_ligand_ff_', ligand_ff)
                        .replace('_lipid_ff_', lipid_ff)
                        .replace('_p_coupling_', p_coupling)
                        .replace('_c_surften_', c_surften)
                        )
                with open(fpath, "w") as f:
                    f.write(s)

        if os.path.exists('run_files'):
            shutil.rmtree('./run_files')
        try:
            shutil.copytree(run_files_orig, './run_files')
        # Directories are the same
        except shutil.Error as e:
            logger.warning('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            logger.warning('Directory not copied. Error: %s' % e)
        if hmr == 'no':
            replacement = 'full.prmtop'
            for dname, dirs, files in os.walk('./run_files'):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.hmr.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)
        elif hmr == 'yes':
            replacement = 'full.hmr.prmtop'
            for dname, dirs, files in os.walk('./run_files'):
                for fname in files:
                    fpath = os.path.join(dname, fname)
                    with open(fpath) as f:
                        s = f.read()
                        s = s.replace('full.prmtop', replacement)
                    with open(fpath, "w") as f:
                        f.write(s)

    # Create window directory
    if not os.path.exists('%s%02d' % (comp, int(win))):
        os.makedirs('%s%02d' % (comp, int(win)))
    os.chdir('%s%02d' % (comp, int(win)))
    # Find already built system in restraint window
    altm = 'None'
    altm_list = ['a00', 'l00', 't00', 'm00']
    if comp == 'a' or comp == 'l' or comp == 't' or comp == 'm':
        for i in altm_list:
            if os.path.exists('../'+i+'/full.hmr.prmtop'):
                altm = i
                break

    if int(win) == 0 and altm == 'None':
        # Build new system
        for file in glob.glob(f'../../{build_file_path}/vac_ligand*'):
            shutil.copy(file, './')
        try:
            shutil.copy(f'../../{build_file_path}/%s.pdb' % mol.lower(), './')
            shutil.copy(f'../../{build_file_path}/fe-%s.pdb' % mol.lower(), './build-ini.pdb')
            shutil.copy(f'../../{build_file_path}/fe-%s.pdb' % mol.lower(), './')
            shutil.copy(f'../../{build_file_path}/anchors-'+pose+'.txt', './')
            shutil.copy(f'../../{build_file_path}/equil-reference.pdb', './')
        except:
            print(os.getcwd())
            raise ValueError('Error copying files')
        for file in glob.glob('../../ff/*.mol2'):
            shutil.copy(file, './')
        for file in glob.glob('../../ff/*.frcmod'):
            shutil.copy(file, './')
        for file in glob.glob('../../ff/%s.*' % mol.lower()):
            shutil.copy(file, './')
        for file in glob.glob('../../ff/dum.*'):
            shutil.copy(file, './')

        # Get TER statements
        ter_atom = []
        with open(f'../../{build_file_path}/rec_file.pdb') as oldfile, open('rec_file-clean.pdb', 'w') as newfile:
            for line in oldfile:
                if not 'WAT' in line:
                    newfile.write(line)
        run_with_log('pdb4amber -i rec_file-clean.pdb -o rec_amber.pdb -y')
        with open('./rec_amber.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'TER'):
                ter_atom.append(int(lines[i][6:11].strip()))

        dum_coords = []
        recep_coords = []
        lig_coords = []
        oth_coords = []
        dum_atomlist = []
        lig_atomlist = []
        recep_atomlist = []
        oth_atomlist = []
        dum_rsnmlist = []
        recep_rsnmlist = []
        lig_rsnmlist = []
        oth_rsnmlist = []
        dum_rsidlist = []
        recep_rsidlist = []
        lig_rsidlist = []
        oth_rsidlist = []
        dum_chainlist = []
        recep_chainlist = []
        lig_chainlist = []
        oth_chainlist = []
        dum_atom = 0
        lig_atom = 0
        recep_atom = 0
        oth_atom = 0
        total_atom = 0
        resid_lig = 0
        resname_lig = self.mol

        # Read coordinates for dummy atoms
        if dec_method == 'sdr' or dec_method == 'exchange':
            for i in range(1, 3):
                shutil.copy(f'../../{build_file_path}/dum'+str(i)+'.pdb', './')
                with open('dum'+str(i)+'.pdb') as dum_in:
                    lines = (line.rstrip() for line in dum_in)
                    lines = list(line for line in lines if line)
                    dum_coords.append((float(lines[1][30:38].strip()), float(
                        lines[1][38:46].strip()), float(lines[1][46:54].strip())))
                    dum_atomlist.append(lines[1][12:16].strip())
                    dum_rsnmlist.append(lines[1][17:20].strip())
                    dum_rsidlist.append(float(lines[1][22:26].strip()))
                    dum_chainlist.append(lines[1][21].strip())
                    dum_atom += 1
                    total_atom += 1
        else:
            for i in range(1, 2):
                shutil.copy(f'../../{build_file_path}/dum'+str(i)+'.pdb', './')
                with open('dum'+str(i)+'.pdb') as dum_in:
                    lines = (line.rstrip() for line in dum_in)
                    lines = list(line for line in lines if line)
                    dum_coords.append((float(lines[1][30:38].strip()), float(
                        lines[1][38:46].strip()), float(lines[1][46:54].strip())))
                    dum_atomlist.append(lines[1][12:16].strip())
                    dum_rsnmlist.append(lines[1][17:20].strip())
                    dum_rsidlist.append(float(lines[1][22:26].strip()))
                    dum_chainlist.append(lines[1][21].strip())
                    dum_atom += 1
                    total_atom += 1

        # Read coordinates from aligned system
        with open('build-ini.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list

        # Count atoms of the system
        for i in range(0, len(lines)):
            if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                molecule = lines[i][17:21].strip() 
                if (molecule != mol) and (molecule != 'DUM') and (molecule != 'WAT') and (molecule not in other_mol) and (molecule not in lipid_mol):
                    recep_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    recep_atomlist.append(lines[i][12:16].strip())
                    recep_rsnmlist.append(molecule)
                    recep_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    recep_chainlist.append(lines[i][21].strip())
                    recep_last = int(lines[i][22:26].strip())
                    recep_atom += 1
                    total_atom += 1
                elif molecule == mol:
                    lig_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    lig_atomlist.append(lines[i][12:16].strip())
                    lig_rsnmlist.append(molecule)
                    lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    lig_chainlist.append(lines[i][21].strip())
                    lig_atom += 1
                    total_atom += 1
                elif (molecule == 'WAT') or (molecule in other_mol):
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1
                elif molecule in lipid_mol:
                    oth_coords.append((float(lines[i][30:38].strip()), float(
                        lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                    oth_atomlist.append(lines[i][12:16].strip())
                    oth_rsnmlist.append(molecule)
                    oth_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                    oth_chainlist.append(lines[i][21].strip())
                    oth_atom += 1
                    total_atom += 1

        coords = dum_coords + recep_coords + lig_coords + oth_coords
        atom_namelist = dum_atomlist + recep_atomlist + lig_atomlist + oth_atomlist
        resid_list = dum_rsidlist + recep_rsidlist + lig_rsidlist + oth_rsidlist
        resname_list = dum_rsnmlist + recep_rsnmlist + lig_rsnmlist + oth_rsnmlist
        chain_list = dum_chainlist + recep_chainlist + lig_chainlist + oth_chainlist
        lig_resid = recep_last + dum_atom
        oth_tmp = 'None'

        # Get coordinates from reference ligand
        if comp == 'x':
            shutil.copy('../../exchange_files/%s.pdb' % molr.lower(), './')
            shutil.copy('../../exchange_files/anchors-'+poser+'.txt', './')
            shutil.copy('../../exchange_files/vac_ligand.pdb', './vac_reference.pdb')
            shutil.copy('../../exchange_files/vac_ligand.prmtop', './vac_reference.prmtop')
            shutil.copy('../../exchange_files/vac_ligand.inpcrd', './vac_reference.inpcrd')
            shutil.copy('../../exchange_files/fe-%s.pdb' % molr.lower(), './build-ref.pdb')

            ref_lig_coords = []
            ref_lig_atomlist = []
            ref_lig_rsnmlist = []
            ref_lig_rsidlist = []
            ref_lig_chainlist = []
            ref_lig_atom = 0
            ref_resid_lig = 0
            resname_lig = molr

            # Read coordinates from reference system
            with open('build-ref.pdb') as f_in:
                lines = (line.rstrip() for line in f_in)
                lines = list(line for line in lines if line)  # Non-blank lines in a list

            # Count atoms of the system
            for i in range(0, len(lines)):
                if (lines[i][0:6].strip() == 'ATOM') or (lines[i][0:6].strip() == 'HETATM'):
                    if lines[i][17:20].strip() == molr:
                        ref_lig_coords.append((float(lines[i][30:38].strip()), float(
                            lines[i][38:46].strip()), float(lines[i][46:54].strip())))
                        ref_lig_atomlist.append(lines[i][12:16].strip())
                        ref_lig_rsnmlist.append(lines[i][17:20].strip())
                        ref_lig_rsidlist.append(float(lines[i][22:26].strip()) + dum_atom - 1)
                        ref_lig_chainlist.append(lines[i][21].strip())
                        ref_lig_atom += 1

        # Write the new pdb file

        build_file = open('build.pdb', 'w')

        # Positions for the dummy atoms
        for i in range(0, dum_atom):
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
            build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        chain_tmp = 'None'
        # Positions of the receptor atoms
        for i in range(dum_atom, dum_atom + recep_atom):
            if chain_list[i] != chain_tmp:
                if resname_list[i] not in other_mol and resname_list[i] != 'WAT':
                    build_file.write('TER\n')
            chain_tmp = chain_list[i]

            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, atom_namelist[i], resname_list[i], chain_list[i], resid_list[i]))
            build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))
            j = i + 2 - dum_atom
            if j in ter_atom:
                build_file.write('TER\n')

        # Positions of the ligand atoms
        for i in range(dum_atom + recep_atom, dum_atom + recep_atom + lig_atom):
            if comp == 'n':
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(coords[i][0]), float(coords[i][1]), float(coords[i][2]+sdr_dist)))
                build_file.write('%6.2f%6.2f\n' % (0, 0))
            elif comp != 'r':
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, atom_namelist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))
                build_file.write('%6.2f%6.2f\n' % (0, 0))

        if comp != 'r':
            build_file.write('TER\n')

        # Extra guests for decoupling

        build_file = open('build.pdb', 'a')
        if (comp == 'e'):
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+1)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

            if dec_method == 'sdr' or dec_method == 'exchange':
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+2)))
                    build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                        lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+3)))
                    build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                        lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
        if comp == 'v' and (dec_method == 'sdr' or dec_method == 'exchange'):
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid + 1)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                    lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        # Other ligands for relative calculations
        if (comp == 'x'):
            for i in range(0, ref_lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, ref_lig_atomlist[i], molr, lig_chainlist[i], float(lig_resid + 1)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(ref_lig_coords[i][0]), float(
                    ref_lig_coords[i][1]), float(ref_lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            for i in range(0, ref_lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, ref_lig_atomlist[i], molr, lig_chainlist[i], float(lig_resid + 2)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(ref_lig_coords[i][0]), float(
                    ref_lig_coords[i][1]), float(ref_lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, lig_chainlist[i], float(lig_resid+3)))
                build_file.write('%8.3f%8.3f%8.3f' % (float(lig_coords[i][0]), float(
                    lig_coords[i][1]), float(lig_coords[i][2]+sdr_dist)))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')

        # Positions of the other atoms
        for i in range(0, oth_atom):
            if oth_rsidlist[i] != oth_tmp:
                build_file.write('TER\n')
            oth_tmp = oth_rsidlist[i]
            build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                             ('ATOM', i+1, oth_atomlist[i], oth_rsnmlist[i], oth_chainlist[i], oth_rsidlist[i]))
            build_file.write('%8.3f%8.3f%8.3f' %
                             (float(oth_coords[i][0]), float(oth_coords[i][1]), float(oth_coords[i][2])))

            build_file.write('%6.2f%6.2f\n' % (0, 0))

        build_file.write('TER\n')
        build_file.write('END\n')
        build_file.close()

        # Write dry build file

        with open('build.pdb') as f_in:
            lines = (line.rstrip() for line in f_in)
            lines = list(line for line in lines if line)  # Non-blank lines in a list
        with open('./build-dry.pdb', 'w') as outfile:
            for i in range(0, len(lines)):
                if lines[i][17:20].strip() == 'WAT':
                    break
                outfile.write(lines[i]+'\n')

        outfile.close()

        if (comp == 'f' or comp == 'w' or comp == 'c'):
            # Create system with one or two ligands
            build_file = open('build.pdb', 'w')
            for i in range(0, lig_atom):
                build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                 ('ATOM', i+1, lig_atomlist[i], mol, chain_list[i], float(lig_resid)))
                build_file.write('%8.3f%8.3f%8.3f' %
                                 (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                build_file.write('%6.2f%6.2f\n' % (0, 0))
            build_file.write('TER\n')
            if comp == 'f':
                for i in range(0, lig_atom):
                    build_file.write('%-4s  %5s %-4s %3s %1s%4.0f    ' %
                                     ('ATOM', i+1, lig_atomlist[i], mol, chain_list[i], float(lig_resid + 1)))
                    build_file.write('%8.3f%8.3f%8.3f' %
                                     (float(lig_coords[i][0]), float(lig_coords[i][1]), float(lig_coords[i][2])))

                    build_file.write('%6.2f%6.2f\n' % (0, 0))
                build_file.write('TER\n')
            build_file.write('END\n')
            build_file.close()
            shutil.copy('./build.pdb', './%s.pdb' % mol.lower())
            tleap_vac = open('tleap_vac.in', 'w')
            tleap_vac.write('source leaprc.'+ligand_ff+'\n\n')
            tleap_vac.write('# Load the ligand parameters\n')
            tleap_vac.write('loadamberparams %s.frcmod\n' % (mol.lower()))
            tleap_vac.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
            tleap_vac.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
            tleap_vac.write('check model\n')
            tleap_vac.write('savepdb model vac.pdb\n')
            tleap_vac.write('saveamberparm model vac.prmtop vac.inpcrd\n')
            tleap_vac.write('quit\n\n')
            tleap_vac.close()

            p = run_with_log(tleap + ' -s -f tleap_vac.in > tleap_vac.log')
    # Copy system from other attach component
    if int(win) == 0 and altm != 'None':
        logger.debug('Copying system from %s' % altm)
        for file in glob.glob('../'+altm+'/*'):
            try:
                shutil.copy(file, './')
            except shutil.Error as e:
                logger.warning('File not copied. Error: %s' % e)
        return 'altm'
    # Copy system initial window
    if win != 0:
        for file in glob.glob('../'+comp+'00/*'):
            try:
                shutil.copy(file, './')
            except shutil.Error as e:
                logger.warning('File not copied. Error: %s' % e)

    return 'all'


def create_box(comp, hmr,
               pose, mol,
               molr, num_waters,
               water_model, ion_def,
               neut, buffer_x, buffer_y, buffer_z,
               stage, ntpr, ntwr, ntwe, ntwx,
               cut, gamma_ln, barostat,
               receptor_ff, ligand_ff,
               dt, dec_method, other_mol, solv_shell,
               lipid_mol, lipid_ff
               ):
    if lipid_mol:
        amber_files_path = 'amber_files'
    else:
        amber_files_path = 'amber_files_no_lipid'

    # Adjust buffers to solvation shell
    if stage == 'fe' and solv_shell != 0:
        buffer_x = buffer_x - solv_shell
        buffer_y = buffer_y - solv_shell
        if buffer_z != 0:
            if ((dec_method == 'sdr') and (comp == 'e' or comp == 'v')) or comp == 'n' or comp == 'x':
                buffer_z = buffer_z - (solv_shell/2)
            else:
                buffer_z = buffer_z - solv_shell

    # Copy and replace simulation files
    if stage != 'fe':
        if os.path.exists(amber_files_path):
            shutil.rmtree(amber_files_path)
        try:
            shutil.copytree(amber_files_orig,
                            amber_files_path)
        # Directories are the same
        except shutil.Error as e:
            logger.warning('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            logger.warning('Directory not copied. Error: %s' % e)
        
        if lipid_mol:
            p_coupling = '3'
            c_surften  = '3'
        else:
            p_coupling = '1'
            c_surften  = '0'
        for dname, dirs, files in os.walk(amber_files_path):
            for fname in files:
                fpath = os.path.join(dname, fname)
                with open(fpath) as f:
                    s = f.read()
                    s = (s
                    .replace('_step_', dt)
                    .replace('_ntpr_', ntpr)
                    .replace('_ntwr_', ntwr)
                    .replace('_ntwe_', ntwe)
                    .replace('_ntwx_', ntwx)
                    .replace('_cutoff_', cut)
                    .replace('_gamma_ln_', gamma_ln)
                    .replace('_barostat_', barostat)
                    .replace('_receptor_ff_', receptor_ff)
                    .replace('_ligand_ff_', ligand_ff)
                    .replace('_lipid_ff_', lipid_ff)
                    .replace('_p_coupling_', p_coupling)
                    .replace('_c_surften_', c_surften)
                    )
                with open(fpath, "w") as f:
                    f.write(s)
        os.chdir(pose)

    # if building a lipid system
    # use x, y box dimensions from the lipid system
    if lipid_mol:
        buffer_x = 0
        buffer_y = 0
        
    # Copy tleap files that are used for restraint generation and analysis
    shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap_vac.in')
    shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap_vac_ligand.in')
    shutil.copy(f'../{amber_files_path}/tleap.in.amber16', 'tleap.in')

    # Copy ligand parameter files
    for file in glob.glob('../ff/*'):
        shutil.copy(file, './')

    # Append tleap file for vacuum
    tleap_vac = open('tleap_vac.in', 'a')
    tleap_vac.write('# Load the necessary parameters\n')
    for i in range(0, len(other_mol)):
        tleap_vac.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
        tleap_vac.write('%s = loadmol2 %s.mol2\n' % (other_mol[i].upper(), other_mol[i].lower()))
    tleap_vac.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_vac.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    if comp == 'x':
        tleap_vac.write('loadamberparams %s.frcmod\n' % (molr.lower()))
    if comp == 'x':
        tleap_vac.write('%s = loadmol2 %s.mol2\n\n' % (molr.upper(), molr.lower()))
    tleap_vac.write('# Load the water parameters\n')
    if water_model.lower() != 'tip3pf':
        tleap_vac.write('source leaprc.water.%s\n\n' % (water_model.lower()))
    else:
        tleap_vac.write('source leaprc.water.fb3\n\n')
    tleap_vac.write('model = loadpdb build-dry.pdb\n\n')
    tleap_vac.write('check model\n')
    tleap_vac.write('savepdb model vac.pdb\n')
    tleap_vac.write('saveamberparm model vac.prmtop vac.inpcrd\n')
    tleap_vac.write('quit\n')
    tleap_vac.close()

    # Append tleap file for ligand only
    tleap_vac_ligand = open('tleap_vac_ligand.in', 'a')
    tleap_vac_ligand.write('# Load the ligand parameters\n')
    tleap_vac_ligand.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_vac_ligand.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    tleap_vac_ligand.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
    tleap_vac_ligand.write('check model\n')
    tleap_vac_ligand.write('savepdb model vac_ligand.pdb\n')
    tleap_vac_ligand.write('saveamberparm model vac_ligand.prmtop vac_ligand.inpcrd\n')
    tleap_vac_ligand.write('quit\n')
    tleap_vac_ligand.close()

    # Generate complex in vacuum
    p = run_with_log(tleap + ' -s -f tleap_vac.in > tleap_vac.log')

    # Generate ligand structure in vacuum
    p = run_with_log(tleap + ' -s -f tleap_vac_ligand.in > tleap_vac_ligand.log')

    # Find out how many cations/anions are needed for neutralization
    neu_cat = 0
    neu_ani = 0
    f = open('tleap_vac.log', 'r')
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip('\'\",.:;#()][')) < 0:
                neu_cat = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
            elif float(splitline[6].strip('\'\",.:;#()][')) > 0:
                neu_ani = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
    f.close()

    # Get ligand removed charge when doing LJ calculations
    lig_cat = 0
    lig_ani = 0
    f = open('tleap_vac_ligand.log', 'r')
    for line in f:
        if "The unperturbed charge of the unit" in line:
            splitline = line.split()
            if float(splitline[6].strip('\'\",.:;#()][')) < 0:
                lig_cat = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
            elif float(splitline[6].strip('\'\",.:;#()][')) > 0:
                lig_ani = round(float(re.sub('[+-]', '', splitline[6].strip('\'\"-,.:;#()]['))))
    f.close()

    # Adjust ions for LJ and electrostatic Calculations (avoid neutralizing plasma)
    if (comp == 'v' and dec_method == 'sdr') or comp == 'x':
        charge_neut = neu_cat - neu_ani - 2*lig_cat + 2*lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)
    if comp == 'e' and dec_method == 'sdr':
        charge_neut = neu_cat - neu_ani - 3*lig_cat + 3*lig_ani
        neu_cat = 0
        neu_ani = 0
        if charge_neut > 0:
            neu_cat = abs(charge_neut)
        if charge_neut < 0:
            neu_ani = abs(charge_neut)

    # Define volume density for different water models
    ratio = 0.060
    if water_model == 'TIP3P':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'SPCE':
        water_box = 'SPCBOX'
    elif water_model == 'TIP4PEW':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'OPC':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'TIP3PF':
        water_box = water_model.upper()+'BOX'

    # Fixed number of water molecules
    if num_waters != 0:

        # Create the first box guess to get the initial number of waters and cross sectional area
        buff = 50.0
        scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
        num_added = scripts.check_tleap()
        cross_area = scripts.cross_sectional_area()

        # First iteration to estimate box volume and number of ions
        res_diff = num_added - num_waters
        buff_diff = res_diff/(ratio*cross_area)
        buff -= buff_diff
        if buff < 0:
            logger.error('Not enough water molecules to fill the system in the z direction, please increase the number of water molecules')
            sys.exit(1)
        # Get box volume and number of added ions
        scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
        box_volume = scripts.box_volume()
        logger.debug(f'Box volume {box_volume}')
        # box volume already takes into account system shrinking during equilibration
        num_cations = round(ion_def[2]*6.02e23*box_volume*1e-27)
        
        # A rough reduction of the number of cations
        # for lipid systems
        if lipid_mol:
            num_cations = num_cations // 2

        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        logger.debug(f'Number of cations: {num_cat}')
        logger.debug(f'Number of anions: {num_ani}')

        # Update target number of residues according to the ion definitions and vacuum waters
        vac_wt = 0
        with open('./build.pdb') as myfile:
            for line in myfile:
                if 'WAT' in line and ' O ' in line:
                    vac_wt += 1
        if (neut == 'no'):
            target_num = int(num_waters - neu_cat + neu_ani + 2*int(num_cations) - vac_wt)
        elif (neut == 'yes'):
            target_num = int(num_waters + neu_cat + neu_ani - vac_wt)

        # Define a few parameters for solvation iteration
        buff = 50.0
        count = 0
        max_count = 10
        rem_limit = 16
        factor = 1
        ind = 0.90
        buff_diff = 1.0

        # Iterate to get the correct number of waters
        while num_added != target_num:
            count += 1
            if count > max_count:
                # Try different parameters
                rem_limit += 4
                if ind > 0.5:
                    ind = ind - 0.02
                else:
                    ind = 0.90
                factor = 1
                max_count = max_count + 10
            tleap_remove = None
            # Manually remove waters if inside removal limit
            if num_added > target_num and (num_added - target_num) < rem_limit:
                difference = num_added - target_num
                tleap_remove = [target_num + 1 + i for i in range(difference)]
                scripts.write_tleap(mol, molr, comp, water_model, water_box, buff,
                                    buffer_x, buffer_y, other_mol, tleap_remove)
                scripts.check_tleap()
                break
            # Set new buffer size based on chosen water density
            res_diff = num_added - target_num - (rem_limit/2)
            buff_diff = res_diff/(ratio*cross_area)
            buff -= (buff_diff * factor)
            if buff < 0:
                logger.error('Not enough water molecules to fill the system in the z direction, please increase the number of water molecules')
                sys.exit(1)
            # Set relaxation factor
            factor = ind * factor
            # Get number of waters
            scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
            num_added = scripts.check_tleap()
        logger.debug(f'{count} iterations for fixed water number')
    # Fixed z buffer
    elif buffer_z != 0:
        buff = buffer_z
        tleap_remove = None
        # Get box volume and number of added ions
        scripts.write_tleap(mol, molr, comp, water_model, water_box, buff, buffer_x, buffer_y, other_mol)
        box_volume = scripts.box_volume()
        logger.debug(f'Box volume {box_volume}')
        # box volume already takes into account system shrinking during equilibration
        num_cations = round(ion_def[2]*6.02e23*box_volume*1e-27)

        # A rough reduction of the number of cations
        # for lipid systems
        if lipid_mol:
            num_cations = num_cations // 2
        # Number of cations and anions
        num_cat = num_cations
        num_ani = num_cations - neu_cat + neu_ani
        # If there are not enough chosen cations to neutralize the system
        if num_ani < 0:
            num_cat = neu_cat
            num_cations = neu_cat
            num_ani = 0
        logger.debug(f'Number of cations: {num_cat}')
        logger.debug(f'Number of anions: {num_ani}')

    # First round just solvate the system
    shutil.copy('tleap.in', 'tleap_solvate_pre.in')
    tleap_solvate = open('tleap_solvate_pre.in', 'a')
    tleap_solvate.write('# Load the necessary parameters\n')
    for i in range(0, len(other_mol)):
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i].upper(), other_mol[i].lower()))
    tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    if comp == 'x':
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
    if comp == 'x':
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr.upper(), molr.lower()))
    tleap_solvate.write('# Load the water and jc ion parameters\n')
    if water_model.lower() != 'tip3pf':
        tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
    else:
        tleap_solvate.write('source leaprc.water.fb3\n\n')
    tleap_solvate.write('model = loadpdb build.pdb\n\n')
    tleap_solvate.write('# Create water box with chosen model\n')
    tleap_solvate.write('solvatebox model ' + water_box +
                        ' {' + str(buffer_x) + ' ' + str(buffer_y) + ' ' + str(buff) + '} 1.5\n\n')
    if tleap_remove is not None:
        tleap_solvate.write('# Remove a few waters manually\n')
        for water in tleap_remove:
            tleap_solvate.write('remove model model.%s\n' % water)
        tleap_solvate.write('\n')
    tleap_solvate.write('desc model\n')
    tleap_solvate.write('savepdb model full_pre.pdb\n')
    tleap_solvate.write('quit')
    tleap_solvate.close()
    p = run_with_log(tleap + ' -s -f tleap_solvate_pre.in > tleap_solvate_pre.log')

    # Retrieve residue number for lipids
    # because tleap separates them into different residues

    run_with_log('pdb4amber -i build.pdb -o build_amber.pdb -y')
    
    renum_data = pd.read_csv('build_amber_renum.txt', sep='\s+',
            header=None, names=['old_resname',
                                'old_chain',
                                'old_resid',
                                'new_resname', 'new_resid'])
    revised_resids = []
    resid_counter = 1
    prev_resid = 0
    for i, row in renum_data.iterrows():
        if row['old_resid'] != prev_resid or row['old_resname'] not in lipid_mol:
            revised_resids.append(resid_counter)
            resid_counter += 1
        else:
            revised_resids.append(resid_counter - 1)
        prev_resid = row['old_resid']
    
    renum_data['revised_resid'] = revised_resids

    try:
        u = mda.Universe('full_pre.pdb')
    except ValueError('could not convert'):
        raise ValueError('The system is toooo big! '
                         'tleap write incorrect PDB when '
                         'residue exceed 100,000.'
                         'I am not sure how to fix it yet.')

    u_orig = mda.Universe('equil-reference.pdb')

    u.dimensions[0] = u_orig.dimensions[0]
    u.dimensions[1] = u_orig.dimensions[1]
    box_xy = [u.dimensions[0], u.dimensions[1]]
    membrane_region = u.select_atoms(f'resname {" ".join(lipid_mol)}')
    # get memb boundries
    membrane_region_z_max = membrane_region.select_atoms('type P').positions[:, 2].max() - 10
    membrane_region_z_min = membrane_region.select_atoms('type P').positions[:, 2].min() + 10
    # water that is within the membrane
    water = u.select_atoms(f'byres (resname WAT and prop z > {membrane_region_z_min} and prop z < {membrane_region_z_max})')

    water_around_prot = u.select_atoms('byres (resname WAT and around 5 protein)')

    final_system = u.atoms - water
    final_system = final_system | water_around_prot

    # get WAT that is out of the box
    outside_wat = final_system.select_atoms(f'byres (resname WAT and ((prop x > {box_xy[0] / 2}) or (prop x < -{box_xy[0] / 2}) or (prop y > {box_xy[1] / 2}) or (prop y < -{box_xy[1] / 2})))')
    final_system = final_system - outside_wat

    logger.debug(f'Final system: {final_system.n_atoms} atoms')

    # set correct residue number
    revised_resids = np.array(revised_resids)
    total_residues = final_system.residues.n_residues
    final_resids = np.zeros(total_residues, dtype=int)
    final_resids[:len(revised_resids)] = revised_resids
    next_resnum = revised_resids[-1] + 1
    final_resids[len(revised_resids):] = np.arange(next_resnum, total_residues - len(revised_resids) + next_resnum)
    final_system.residues.resids = final_resids

    
    final_system_prot = final_system.select_atoms('protein')
    final_system_dum = final_system.select_atoms('resname DUM')
    final_system_others = final_system - final_system_prot - final_system_dum

    for residue in u.select_atoms('protein').residues:
        resid_str = residue.resid
        residue.atoms.chainIDs = renum_data.query(f'old_resid == @resid_str').old_chain.values[0]

    dum_lines = []
    for chain_name in np.unique(final_system.select_atoms('protein').atoms.chainIDs):
        temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

        prot_segment = final_system.select_atoms(f'chainID {chain_name}')

        prot_segment.write(temp_pdb.name)
        temp_pdb.close()

        with open(temp_pdb.name, 'r') as f:
            # store lines start with ATOM
            dum_lines += [line for line in f.readlines() if line.startswith('ATOM')]
        dum_lines.append('TER\n')
    
    with open('solvate_pre_prot.pdb', 'w') as f:
        f.writelines(dum_lines)
    
#    final_system_prot.write('solvate_pre_prot.pdb')

    dum_lines = []
    for residue in final_system_dum.residues:
        # create a temp pdb file in /tmp/
        # write the residue to the temp file
        temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

        residue.atoms.write(temp_pdb.name)
        temp_pdb.close()
        # store atom lines into dum_lines
        with open(temp_pdb.name, 'r') as f:
            # store lines start with ATOM
            dum_lines += [line for line in f.readlines() if line.startswith('ATOM')]
        dum_lines.append('TER\n')
    
    with open('solvate_pre_dum.pdb', 'w') as f:
        f.writelines(dum_lines)

    non_prot_lines = []

    prev_resid = final_system_others.residues.resids[0]
    for residue in final_system_others.residues:
        if residue.resid != prev_resid:
            non_prot_lines.append('TER\n')
        # create a temp pdb file in /tmp/
        # write the residue to the temp file
        temp_pdb = tempfile.NamedTemporaryFile(delete=False, dir='/tmp/', suffix='.pdb')

        residue.atoms.write(temp_pdb.name)
        temp_pdb.close()
        # store atom lines into non_prot_lines
        with open(temp_pdb.name, 'r') as f:
            # store lines start with ATOM
            non_prot_lines += [line for line in f.readlines() if line.startswith('ATOM')]
        prev_resid = residue.resid

    with open('solvate_pre_others.pdb', 'w') as f:
        f.writelines(non_prot_lines)

    box_final = final_system_prot.dimensions
    logger.debug(f'Final box dimensions: {box_final[:3]}')
    # Write the final tleap file with the correct system size and removed water molecules
    shutil.copy('tleap.in', 'tleap_solvate.in')
    tleap_solvate = open('tleap_solvate.in', 'a')
    tleap_solvate.write('# Load the necessary parameters\n')
    for i in range(0, len(other_mol)):
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (other_mol[i].lower()))
        tleap_solvate.write('%s = loadmol2 %s.mol2\n' % (other_mol[i].upper(), other_mol[i].lower()))
    tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    if comp == 'x':
        tleap_solvate.write('loadamberparams %s.frcmod\n' % (molr.lower()))
    if comp == 'x':
        tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (molr.upper(), molr.lower()))
    tleap_solvate.write('# Load the water and jc ion parameters\n')
    if water_model.lower() != 'tip3pf':
        tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
    else:
        tleap_solvate.write('source leaprc.water.fb3\n\n')

    tleap_solvate.write('dum = loadpdb solvate_pre_dum.pdb\n\n')
    tleap_solvate.write('prot = loadpdb solvate_pre_prot.pdb\n\n')
    tleap_solvate.write('others = loadpdb solvate_pre_others.pdb\n\n')
    tleap_solvate.write('model = combine {dum prot others}\n\n')

    if (neut == 'no'):
        tleap_solvate.write('# Add ions for neutralization/ionization\n')
        tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[0], num_cat))
        tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[1], num_ani))
    elif (neut == 'yes'):
        tleap_solvate.write('# Add ions for neutralization/ionization\n')
        if neu_cat != 0:
            tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[0], neu_cat))
        if neu_ani != 0:
            tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[1], neu_ani))
    tleap_solvate.write('\n')
    tleap_solvate.write('set model box {%.2f %.2f %.2f}\n' % (box_final[0], box_final[1], box_final[2]))
    tleap_solvate.write('desc model\n')
    tleap_solvate.write('savepdb model full.pdb\n')
    tleap_solvate.write('saveamberparm model full.prmtop full.inpcrd\n')
    tleap_solvate.write('quit')
    tleap_solvate.close()
    p = run_with_log(tleap + ' -s -f tleap_solvate.in > tleap_solvate.log')

    f = open('tleap_solvate.log', 'r')
    for line in f:
        if "Could not open file" in line:
            logger.error('Error!!!')
            logger.error(line)
            sys.exit(1)
        if "WARNING: The unperturbed charge of the unit:" in line:
            logger.warning(line)
            logger.warning('The system is not neutralized properly after solvation')
        if "addIonsRand: Argument #2 is type String must be of type: [unit]" in line:
            logger.error('Aborted.The ion types specified in the input file could be wrong.')
            logger.error('Please check the tleap_solvate.log file, and the ion types specified in the input file.\n')
            sys.exit(1)
    f.close()

    # Apply hydrogen mass repartitioning
    shutil.copy(f'../{amber_files_path}/parmed-hmr.in', './')
    run_with_log('parmed -O -n -i parmed-hmr.in > parmed-hmr.log')

    if stage != 'fe':
        os.chdir('../')


def ligand_box(mol, lig_buffer, water_model, neut, ion_def, comp, ligand_ff):
    # Define volume density for different water models
    if water_model == 'TIP3P':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'SPCE':
        water_box = 'SPCBOX'
    elif water_model == 'TIP4PEW':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'OPC':
        water_box = water_model.upper()+'BOX'
    elif water_model == 'TIP3PF':
        water_box = water_model.upper()+'BOX'

    # Copy ligand parameter files
    for file in glob.glob('../../ff/%s.*' % mol.lower()):
        shutil.copy(file, './')

    # Write and run preliminary tleap file
    tleap_solvate = open('tmp_tleap.in', 'w')
    tleap_solvate.write('source leaprc.'+ligand_ff+'\n\n')
    tleap_solvate.write('# Load the ligand parameters\n')
    tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    tleap_solvate.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
    tleap_solvate.write('# Load the water and jc ion parameters\n')
    if water_model.lower() != 'tip3pf':
        tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
    else:
        tleap_solvate.write('source leaprc.water.fb3\n\n')
    tleap_solvate.write('check model\n')
    tleap_solvate.write('savepdb model vac.pdb\n')
    tleap_solvate.write('saveamberparm model vac.prmtop vac.inpcrd\n\n')
    tleap_solvate.write('# Create water box with chosen model\n')
    tleap_solvate.write('solvatebox model ' + water_box + ' '+str(lig_buffer)+'\n\n')
    tleap_solvate.write('quit\n')
    tleap_solvate.close()

    # Get box volume and number of added ions
    box_volume = scripts.box_volume()
    logger.debug(f'Box volume: {box_volume}')
    # box volume already takes into account system shrinking during equilibration
    num_cations = round(ion_def[2] * 6.02e23 * box_volume * 1e-27)
    logger.debug(f'Number of cations: {num_cations}')

    # Write and run tleap file
    tleap_solvate = open('tleap_solvate.in', 'a')
    tleap_solvate.write('source leaprc.'+ligand_ff+'\n\n')
    tleap_solvate.write('# Load the ligand parameters\n')
    tleap_solvate.write('loadamberparams %s.frcmod\n' % (mol.lower()))
    tleap_solvate.write('%s = loadmol2 %s.mol2\n\n' % (mol.upper(), mol.lower()))
    tleap_solvate.write('model = loadpdb %s.pdb\n\n' % (mol.lower()))
    tleap_solvate.write('# Load the water and jc ion parameters\n')
    if water_model.lower() != 'tip3pf':
        tleap_solvate.write('source leaprc.water.%s\n\n' % (water_model.lower()))
    else:
        tleap_solvate.write('source leaprc.water.fb3\n\n')
    tleap_solvate.write('check model\n')
    tleap_solvate.write('savepdb model vac.pdb\n')
    tleap_solvate.write('saveamberparm model vac.prmtop vac.inpcrd\n\n')
    tleap_solvate.write('# Create water box with chosen model\n')
    tleap_solvate.write('solvatebox model ' + water_box + ' '+str(lig_buffer)+' 1.5\n\n')
    if (neut == 'no'):
        tleap_solvate.write('# Add ions for neutralization/ionization\n')
        tleap_solvate.write('addionsrand model %s %d\n' % (ion_def[0], num_cations))
        tleap_solvate.write('addionsrand model %s 0\n' % (ion_def[1]))
    elif (neut == 'yes'):
        tleap_solvate.write('# Add ions for neutralization/ionization\n')
        tleap_solvate.write('addionsrand model %s 0\n' % (ion_def[0]))
        tleap_solvate.write('addionsrand model %s 0\n' % (ion_def[1]))
    tleap_solvate.write('\n')
    tleap_solvate.write('desc model\n')
    tleap_solvate.write('savepdb model full.pdb\n')
    tleap_solvate.write('saveamberparm model full.prmtop full.inpcrd\n')
    tleap_solvate.write('quit\n')
    tleap_solvate.close()
    p = run_with_log(tleap + ' -s -f tleap_solvate.in > tleap_solvate.log')

    # Apply hydrogen mass repartitioning
    shutil.copy('../amber_files_no_lipid/parmed-hmr.in', './')
    run_with_log('parmed -O -n -i parmed-hmr.in > parmed-hmr.log')

    # Copy a few files for consistency
    if (comp != 'f' and comp != 'w'):
        shutil.copy('./vac.pdb', './vac_ligand.pdb')
        shutil.copy('./vac.prmtop', './vac_ligand.prmtop')
