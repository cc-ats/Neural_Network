#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:19:17 2017

Used to take a set of .xyz files and associated QCHEM energy calculations
and convert them into usable inputs (binarized coulomb matrices) for a
neural network, and to associate those inputs with the calculated 
QM-MM energy difference.

@author: johnalberse
"""

#import tensorFlow as tf
import numpy as np
import math
import glob
import periodictable as pt
import random


#TODO: Set this when I finalize the data set and/or begin work on architecture
COULOMB_DIMENSION = 23

#TODO: Make this file conform to python naming conventions and practices

#TODO: test
def generateGeometryEnergyPairs():
    """
    Takes all the xyz files and all the energy calculations
    and generates a list of tuples
        x = tuple[0] == binarized geometry
        y = tuple[1] == calculated energy difference (QM - MM)
    
    This list will be split into training, validation, and testing sets later
    
    data should be a folder containing all xyz files and a single
    energy file.
    xyz file names MUST match the associated labels on each line of energy file
    """
    #create ges, a list of tuples (binarized geometry, energy difference)
    ges = []
    try:
        molecules = open("data/energy.txt", "r").readlines()
        #using this to count num of molecules, so we can divide into datasets
        numMolecules = 0
        #loop through lines in energy.txt
        for molecule in molecules:
            numMolecules = numMolecules + 1
            m = molecule.split()
            # m[0] == xyz file name, m[1] == num atoms, m[2] == MM, m[3] == QM
            #use first element of each line to find associated xyz file
            #Construct coulomb matrix
            c = coulombConstructor("data/" + m[0] + ".xyz")
            #Get the energy difference e = (QM - MM)
            e = float(m[3]) - float(m[2])
        
            #generate 10 random matrices and append their binarized forms
            for i in range (0, 10):
                rc = randomSort(c)
                #note that we flatten binaries here
                bc = binarizeMatrix(rc).flatten()
                ges.append( (bc, e) )
    
        #shuffle ges, so splitting data sets will be truly random
        random.shuffle(ges)
        
        #TODO: Write to 3 files: training, validation, and testing
        #maybe just training and validaiton for now
        
        
        
        #write the data to a file
        with open( 'data/data.txt', 'w' ) as f:
            np.set_printoptions(threshold=np.inf)
            f.write('\n'.join('%s %s' % x for x in ges))
            f.close()
    except FileNotFoundError:
        #might pass this error up later instead
        print("data folder or datafolder/energies.txt is missing")
        return None
    
    #return ges


def formatInput(xyzFile):
    """
    Formats an xyzFile input into a suitable sensory input AT RUN TIME
    
    Gets 10 associated random coulomb matrices, averages them, and then
    binarizes them for input
    """
    c = coulombConstructor(xyzFile)
    rcs = []
    for i in range(0, 10):
        rcs.append(randomSort(c))
    rc = np.mean(np.array(rcs), axis=0)
    
    nnInput = binarizeMatrix(rc)
    
    return nnInput
        

#TODO: Figure out units. Angstroms? Check QCHEM output files
#TODO: Refactor as generateCoulombMatrix to keep with naming convention
def coulombConstructor(xyzFile):
    """
    Based on Coulomb construction from
    "Learning Invariant Represntations of Molecules for Atomization
    Energy Prediction" by Machine Learning group, TU Berlin
    In reference to work done by Rupp
    "Fast and Accurate Modeling of Molecular 
    Atomization Energies with Machine Learning"
    
    z - a python list of nuclear charges
    r - Corresponding python list of cartesian coordinates of
        molecular positions in 3 space
    
    Constructs and returns coulomb matrix c
    The base of a learning invariant representation of a molecule
    
    xyzFile: The file from xyzFiles to make c for
    d: the size of c, previously obtained through getCoulombDimension
    """
    # makes rz from the xyz file
    rz = makeRZ(xyzFile)
    r = rz[0]
    z = rz[1]
    
    #c, a numpy array with each dimension of size d
    c = np.zeros( (COULOMB_DIMENSION,COULOMB_DIMENSION) )
    #current contents are just a dxd matrix of 0s
    
    
    # i is row
    # j is column
    # set as specified
    for i in range(0, COULOMB_DIMENSION):
        for j in range(0, COULOMB_DIMENSION):
            #only if i and j are valid indices in r and z
            if i < len(z) and j < len(z) and i < len(r) and j < len(r):
                if (i == j):
                    c[i,j] = .5 * math.pow(z[i], 2.4)
                else:
                    c[i,j] = ((z[i] * z[j]) / 
                     np.linalg.norm(np.subtract(r[i], r[j])))
            #rest is already "0 molecules"
    
    return c


def makeRZ(xyzFile):
    """
    xyzFile: file path to .xyz file, 
    Which contains the number of atoms on the first line,
    nothing on the second,
    and then one atom per line - atomic symbol followed by x y z values
    
    Returns the Z and R values for coulomb constructor as a tuple of lists
    z is an int (atomic number), r is a list of 3 floats
    """
    #open file for reading
    f = open(xyzFile, "r")
    
    xyzContents = f.readlines()
    
    #remove first line and assign to numAtoms
    numAtoms  = xyzContents.pop(0)
    #parse to int
    numAtoms = int(numAtoms)
    #remove second line, which is empty
    xyzContents.pop(0)
    
    #creates arrays
    r = []
    zSymbols = []
    for line in xyzContents:
        #grabs the atom from the line
        atom = line.split()
        zSymbols.append(atom.pop(0))
        #creates coords to add to r
        coords = []
        for c in atom:
            coords.append(float(c))
        r.append(coords)
        
    #Take symbols and make convert to atomic numbers
    z = []
    for symbol in zSymbols:
        z.append(pt.elements.symbol(symbol).number)
    
    #closes xyz file
    f.close
    
    #return r, z tuple
    return (r, z)

#TODO: We might be able to trim more constant inputs (i.e. top left 4x4?)
#TODO: Check this is still best on final dataset. May be non-constant longer
def binarizeMatrix(c):
    """
    'break apart each dimension of the Coulomb matrix C by converting the 
    representation into a three-dimensional tensor of essentially 
    binary predicates'
    
    This is similar to image binarization, or thresholding.
    
    Based on work in Machine Learning Group TU Berlin's
    "Learning invariant representations of molecules for
    atomization energy prediction"
    """
    #step. choose to keep tractable
    theta = 1
    
    thetaMatrix = np.full([COULOMB_DIMENSION, COULOMB_DIMENSION], theta)
    x = []
    #change range to get more/less inputs
    #-5 to 0 was chosen because after that, each dimension of the binary
    #tensor tends to become identical to the previous dimension
    #i.e. i = 1, 2, 3... are just 23x23 matrices of 1s
    #i < -5 are relatively constant matrices of both 1s and 0s,
    #likely because the numbers associated with the 1s are too large to be
    # effected by the step within a reasonable number of steps
    #in theory, going further into negatives until we get all 0s would be more
    # accurate, however this is an intractable number of inputs.
    #       if we want to do this, increase step size.
    for i in range(-5, 0):
        x.append(np.tanh(np.divide(np.add(c, i * thetaMatrix), thetaMatrix)))
    #convert x into a numpy array for calculations
    npx = np.array(x).flatten()
    
    #NOTE: below isn't explicitly called for in Berlin's paper, but a practice 
    #borrowed from image thresholding. 
    # Not sure it will work best for this purpose.
    #TODO: If really can't train NN well, reexamine this
    
    #x now has values ranging from -1 to 1. 
    #split on a threshold T, where a = 0 if a < T, a = 1 if a >= T
    T = 0
    bcB = npx >= T
    bc = bcB.astype(int)

    
    #returns the binary matrices
    return bc

def randomSort(c):
    """
    Takes a coulumb matrix c and returns a ONE of its associated "random" 
    coulomb matrices
    
    Based on work in Machine Learning Group TU Berlin's
    "Learning invariant representations of molecules for
    atomization energy prediction"
    
    The purpose of these is to artificially expand the training data set
    by creating more data from what we already have. This improves accuracy.
    """
    #noise generator, change to get different random distributions
    sigma = 1
    
    #compute the row norms of C
    cRowNorms = np.linalg.norm(c, axis=1)
    #add random number 0-1 to each row norm
    randRowNorms = list()
    for rowNorm in cRowNorms:
        #n is assigned n ~ U[0,sigma * 1)
        #Add random number to the row norm and append to new list
        randRowNorms.append(rowNorm + random.random() * sigma)
    #Each row's index is analagous to its randRowNorm's index
    #We must sort the actual rows of c based on randRowNorm descending sort
    #we use argsort to get a list of indices that would sort randRowNorm
    #and we use that to sort the rows of c
    rowIndices = np.argsort(randRowNorms)
    #empty numPy matrix we can dump the correct data into
    permuteRowC = np.zeros( (COULOMB_DIMENSION, COULOMB_DIMENSION) )
    
    #for indexing where we are in rowIndices
    for i in range(0, COULOMB_DIMENSION):
        permuteRowC[i] = c[rowIndices[i]]
        
    #permuteRowC is now a permuted coulomb matrix with randomnly sorted
    #rows. We must do the same permutation to columns to get a fully randomC
    
    #gets column norms of the permuted-row matrix
    cColumnNorms = np.linalg.norm(permuteRowC, axis=0)
    #get new, semi-random row norms
    randColumnNorms = list()
    for columnNorm in cColumnNorms:
        #n is assigned n ~ [0, sigma * 1)
        #appends (column norm + n) to the new list of semi-random norms
        randColumnNorms.append(columnNorm + random.random() * sigma)
    #indices that will sort the columns
    columnIndices = np.argsort(randColumnNorms)
    #empty random matrix of correct dimensions
    cRandom= np.zeros( (COULOMB_DIMENSION, COULOMB_DIMENSION) )
    
    #sorts columns by sorting each row
    for i in range(0, COULOMB_DIMENSION):
        #sort columns in this row
        for j in range(0, COULOMB_DIMENSION):
            cRandom[i,j] = permuteRowC[i, columnIndices[j]]
    
    #flips cRandom to be properly oriented
    cRandom = np.fliplr(cRandom)
    cRandom = np.flipud(cRandom)
    
    #returns the final, randomnly sorted coloumb matrix
    return cRandom
    


#TODO: Use this & manually set size once I decide on a data set
def getCoulombDimension():
    """
    This function takes a list of all the geometries in the input set and finds
    the one with the most atoms
    
    The greatest number of atoms is returned. This is to find a standard
    dimension/size for our coulomb matrix, so that our NN can take in
    a standard number of inputs. 
    
    NOTE: This is a utility function to help determine the architecture of the
    neural network. The coulomb dimension should not change after construction
    of the NN.
    """
    max = 0
    #loops through all files in directory and returns biggest num of atoms
    for filename in glob.glob("data/*.xyz"):
        f = open("filename", "r")
        currentSize = int(f.readline)
        if currentSize > max:
            max = currentSize
    return max

def countEqualElements(ges):
    """
    Utility function. Lets us count the number of inputs in binarized matrices
    that are the same, to make sure there's actually enough difference to 
    allow the NN to work
    
    Also counts the numbe of occurances of them
    """
    cnts = []
    for j in range(0, len(ges)):
        cnt = 0
        for i in range(0, len(ges[0][0])):
            if ges[0][0][i] == ges[j][0][i]:
                cnt = cnt + 1
        if cnt == 2645:
            print(j)
        cnts.append(cnt)
    occurances = []
    for i in range(min(cnts), max(cnts) + 1):
        occurances.append( (i, cnts.count(i)) )
    return occurances

def exampleChain(xyzFile):
    """
    Prints all steps of getting an input to a file, for demonstration purposes
    """
    f = open('example_input.txt', 'w')
    f2 = open(xyzFile, 'r')
    for line in f2.readlines():
        f.write(line)
    f2.close()
    np.set_printoptions(threshold=np.inf)
    f.write('\n\n')
    c = coulombConstructor(xyzFile)
    f.write(np.array2string(c))
    f.write('\n\n')
    rc = randomSort(c)
    f.write(np.array2string(rc))
    f.write('\n\n')
    bc = binarizeMatrix(rc)
    f.write(np.array2string(bc))
    f.close()