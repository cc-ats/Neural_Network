The purpose of this project is to create a deep neural network for the calculation of the 
atomization energies of molecules. This project is to be implemented into the QCHEM interface
for rapid, accurate calculations of atomization energies. 

NN SPECIFICATIONS:
Input: Binarized coulomb matrices. User only has to provide a .xyz file

Output: QM calculation. The result is a much quicker way of approximating the QM value. 
All initial energy calculations done with QCHEM.

TODO:
TRAIN the DNN
	Trained to ~2% error over validation set
IMPROVE the DNN
	Add dropout
	Utilize testing dataset
IMPLEMENT the network into the QCHEM package