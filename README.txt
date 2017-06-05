The purpose of this project is to create a deep neural network for the calculation of the atomization energies of molecules. This project is to be implemented into the QCHEM interface for rapid, accurate calculations of atomization energies. 

NN SPECIFICATIONS:
Input: Binarized coulomb matrices. User only has to provide a .xyz file

Output: QM calculation. The result is a much quicker way of approximating the QM value. All initial energy calculations done with QCHEM.

TODO:
ASSEMBLE databases
	Run calculations for testing, validation data
	TEST if lists are correct, in that they are 1:1 on assembly
TRAIN simple model
CONSTRUCT the DNN - Which will be better than the simple model :)
TRAIN the DNN
IMPLEMENT the network into the QCHEM package