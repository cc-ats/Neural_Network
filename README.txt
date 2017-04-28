The purpose of this project is to create a deep neural network for the calculation of the atomization energies of molecules. This project is to be implemented into the QCHEM interface for rapid, accurate calculations of atomization energies. 

NN SPECIFICATIONS:
Input: Binarized coulomb matrices. User only has to provide a .xyz file

Output: Energy difference, QM-MM, where QM is the quantum mechanical calculation and MM is the less accurate, molecular mechanical calculation. This is a corrective term, to be added to an existing MM calculation. The result is a much quicker way of approximating the QM value. All energy calculations done with QCHEM.

TODO:
ASSEMBLE databases
	GDB13 is extremely expensive to convert as a whole (way to big)
	BUT we can represent parts as individual .smi pretty easily, 
	and convert to xyz as needed. 
	This would be best for testing and validation sets. For training set,
	Converting up-front is probably best (currently in process)
	
	For now, test within pre-converted 10k
CONSTRUCT the DNN
TRAIN the DNN
IMPLEMENT the network into the QCHEM package