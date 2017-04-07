The purpose of this project is to create a deep neural network for the calculation of the atomization energies of molecules. This project is to be implemented into the QCHEM interface for rapid, accurate calculations of atomization energies. 

NN SPECIFICATIONS:
Input: Binarized coulomb matrices. User only has to provide a .xyz file

Output: Energy difference, QM-MM, where QM is the quantum mechanical calculation and MM is the less accurate, molecular mechanical calculation. This is a corrective term, to be added to an existing MM calculation. The result is a much quicker way of approximating the QM value. All energy calculations done with QCHEM.

TODO:
ASSEMBLE databases
CONSTRUCT the DNN
TRAIN the DNN
IMPLEMENT the network into the QCHEM package