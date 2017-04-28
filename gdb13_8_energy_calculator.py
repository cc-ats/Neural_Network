# -*- coding: utf-8 -*-
#
# Copyright (c), the QMAnalysis development team
#
#
#

""" Input Generator """

import numpy as np, sys, os, string
sys.path.insert(0, '../../QMAnalysis/src')
from math_util import *
from qchem_input import *
from qchem_out_parser import QChemOut
from xyz_file_parser import * 
from plot_util import *

command = 'ls -1 ../data/gdb13_8/*.xyz > xyzfilelist'
os.system(command)

xyzfiles = open("xyzfilelist", "r").readlines()
nfiles = len(xyzfiles)
nw = []  # number of water molecules
e1 = []
e2 = []

enf = open("energy.txt", "w")
for k in range(0, nfiles):

	name = xyzfiles[k][14:-5]
	outfile1 = name+".hf.631gd.out"
	out1 = QChemOut()
	out1.setup(outfile1)
	energy1 = out1.scf_energy * 627.5095
	nwater = out1.natoms/3
	nw.append(nwater)
	e1.append(energy1)

	outfile2 = name+".wb97xd.6311pgdp.out"
	out2 = QChemOut()
	out2.setup(outfile2)
	energy2 = out2.scf_energy * 627.5095
	e2.append(energy2)

	enf.write("%30s %3d %17.7f %15.7f\n" % (name, nwater, energy1, energy2))

for n in range(5, 8):
	v1 = np.zeros(nfiles)
	v2 = np.zeros(nfiles)
	count = 0
	for k in range(0, nfiles):
		if nw[k] == n:
			v1[count] = e1[k]
			v2[count] = e2[k]
			#print 'n=', n, 'k=', k, nw[k]
			count += 1

	#print 'v1=', v1[0:count]
	#print 'v2=', v2[0:count]
	np.scatter_plot(v1[0:count], v2[0:count], "HF/6-31+G*", "wB97X-D/6-311+G**", "Water-Cluster-HF-vs-wB97XD."+str(n)+"mer.png")