The purpose of this project is to create a deep neural network for the calculation of the atomization energies of molecules. This project is to be implemented into the QCHEM interface for rapid, accurate calculations of atomization energies. 

NN SPECIFICATIONS:
Input: Binarized coulomb matrices. User only has to provide a .xyz file

Output: QM calculation. The result is a much quicker way of approximating the QM value. All initial energy calculations done with QCHEM.

TODO:
ASSEMBLE databases
	GDB13 partially converted to .xyz (only 8.smi). These need to have QM calculations run on them. With this data we can attempt to train simple model.
	We can also work to create a lot of water cluster data and calculate those
TRAIN simple model
CONSTRUCT the DNN - Which will be better than the simple model :)
TRAIN the DNN
IMPLEMENT the network into the QCHEM package