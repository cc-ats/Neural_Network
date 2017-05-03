import numpy as np
import glob 
import sys, os
import os.path
from qm_analysis import *

subset = int(sys.argv[1])
energyfile = 'energies.batch'+str(subset)+".dat"

options  = {'method':'b3lyp', 'dft_d':'empirical_grimme3', 'basis':'6-31G*'}
rem = QChemRem(options)

count = 0
previous_energies = {}  # energies already computed
if os.path.isfile(energyfile):
	datf = open(energyfile, "r")
	for line in datf.readlines():
		cols = line.split()
		previous_energies[cols[0]] = float(cols[1])
	print("previous_energies:", previous_energies)
	enf = open(energyfile, "a")
else:
	enf = open(energyfile, "w")

for xyzfile in glob.glob("data/gdb_subset/training/"+"mol"+str(subset)+"*xyz"):
	pos1  = xyzfile.find("mol")+3
	index = int(xyzfile[pos1:-4])
	if str(index) in previous_energies.keys():
		print(index, "already computed")
		continue

	molecule = read_xyz(xyzfile)
	print("xyzfile: ", index, xyzfile, index, molecule.natoms)
	qchem_input_generator("qchem.inp", molecule, rem)	
	#os.system("qchem -nt 20 qchem.inp qchem.out")
	os.system("qchem qchem.inp qchem.out")

	qcout = QChemOut("qchem.out")
	energy = qcout.scf_energy
	enf.write("%10d %20.10f \n" % (index, energy))

	count += 1
	if count > 4: break
	

