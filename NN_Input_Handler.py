#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 10:19:17 2017

Used to take a set of .xyz files and associated QCHEM energy calculations
and convert them into usable inputs (binarized semi-random coulomb matrices)
for neural network, and to associate those inputs with the calculated 
QM energies. Generates tfrecords files

@author: johnalberse
"""

import tensorflow as tf
import numpy as np
import math
import glob
import periodictable as pt
import random
import os
import random as rd

#TODO: Set this when I finalize the data set and/or begin work on architecture
COULOMB_DIMENSION = 23

#TODO: retest for new implementation once energies calculated/implemented
def generate_geometry_energy_pairs():
    """
    Generates coulomb energy pairs and writes in serialized tfrecords format
    """
    try:
        #For each dataset, go through associated energies file and append
        #mol/energy to appropriate lists based on ID in energies file
        #assigning in this way to each list assures 1:1 correlation
        #for easy pairing later
        
        #a list of files for each mol
        training_molecules = []
        #those mol's energies
        training_energies = []
        training_energies_file = 'data/gdb_subset/training/training_energies'
        with open(training_energies_file, 'r') as f:
            for line in f:
                dat = line.split()
                training_molecules.append(
                        'data/gdb_subset/training/mol' + dat[0] + '.xyz')
                training_energies.append(float(dat[1]))
        
        validation_molecules = []
        validation_energies = []
        validation_energies_file = 'data/gdb_subset/validation/validation_energies'
        with open(validation_energies_file, 'r') as f:
            for line in f:
                dat = line.split()
                validation_molecules.append(
                        'data/gdb_subset/validation/mol' + dat[0] + '.xyz')
                validation_energies.append(float(dat[1]))
        
        testing_molecules = []
        testing_energies = []
        testing_energies_file = 'data/gdb_subset/testing/testing_energies'
        with open(testing_energies_file, 'r') as f:
            for line in f:
                dat = line.split()
                testing_molecules.append(
                        'data/gdb_subset/testing/mol' + dat[0] + '.xyz')
                testing_energies.append(float(dat[1]))
        
        #associates energies and molecules, writes to tfrecords
        write_to_files(training_molecules, training_energies,
                       validation_molecules, validation_energies,
                       testing_molecules, testing_energies)
        
    except FileNotFoundError:
        print("data or energy calculations are missing from data folder")

def write_to_files(training_molecules, training_energies,
                   validation_molecules, validation_energies,
                   testing_molecules, testing_energies):
    """
    Takes molecules and energies lists, and writes them to tfRecords file
    for input into neural network
    """
    #writer for each input (for the neural network) file
    training_writer = tf.python_io.TFRecordWriter("training.tfrecords")
    testing_writer = tf.python_io.TFRecordWriter("testing.tfrecords")
    validation_writer = tf.python_io.TFRecordWriter("validation.tfrecords")
    
    for i in range(len(training_molecules)):
        c = generateCoulombMatrix(training_molecules[i])
        for i in range (0, 10):
            training_writer.write(
                    get_serialized_example(c, training_energies[i]))
    for i in range(len(validation_molecules)):
        c = generateCoulombMatrix(validation_molecules[i])
        for i in range (0, 10):
            validation_writer.write(
                    get_serialized_example(c, validation_energies[i]))
    
    for i in range(len(testing_molecules)):
        c = generateCoulombMatrix(testing_molecules[i])
        for i in range (0, 10):
            testing_writer.write(
                    get_serialized_example(c, testing_energies[i]))

def get_serialized_example(c, e):
    """
    Returns a serialized example to write into the tfrecords file
    
    Includes conversion into random binzarized rank 1 matrix
    """
    rc = random_sort(c)
    bc = binarize_matrix(rc)
    example= tf.train.Example(
        #example contains a proto Features object
        features=tf.train.Features(
            #Features contains a proto map of string to Features
            feature={
                    'bc': tf.train.Feature(
                            float_list=tf.train.FloatList(value=bc)),
                    'e': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[e]))
            }))
    #use the proto object to serialize exaple to string
    return example.SerializeToString()

def format_input(xyzFile):
    """
    Formats an xyzFile input into a suitable sensory input AT RUN TIME
    
    Gets 10 associated random coulomb matrices, averages them, and then
    binarizes them for input
    """
    c = generateCoulombMatrix(xyzFile)
    rcs = []
    for i in range(0, 10):
        rcs.append(random_sort(c))
    rc = np.mean(np.array(rcs), axis=0)
    
    nnInput = binarize_matrix(rc)
    
    return nnInput

#TODO: Figure out units. Angstroms? Check QCHEM output files
#TODO: Refactor as generateCoulombMatrix to keep with naming convention
def generateCoulombMatrix(xyzFile):
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
    rz = make_rz(xyzFile)
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


def make_rz(xyzFile):
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
#TODO: RECHECK step/num of matrices EVERY TIME DATASET CHANGES. 
#TODO: Changing this should be one of the first steps in improving network
def binarize_matrix(c):
    """
    'break apart each dimension of the Coulomb matrix C by converting the 
    representation into a three-dimensional tensor of essentially 
    binary predicates'
    
    Based on work in Machine Learning Group TU Berlin's
    "Learning invariant representations of molecules for
    atomization energy prediction"
    
    Also incorporates ideas from image thresholding
    """
    #step. choose to keep tractable
    theta = 1
    
    thetaMatrix = np.full([COULOMB_DIMENSION, COULOMB_DIMENSION], theta)
    x = []
    #change range to get more/less inputs
    #-5 to 0 chosen because outside of this range each dimension of the matrix
    #remains relatively constant
    for i in range(-5, 0):
        x.append(np.tanh(np.divide(np.add(c, i * thetaMatrix), thetaMatrix)))
    #convert x into a numpy array for calculations
    npx = np.array(x).flatten()
    
    #thresholding
    T = 0
    bcB = npx >= T
    bc = bcB.astype(int)

    #returns the binary matrices
    return bc

def random_sort(c):
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
    


#TODO: Use this & manually set size every time new dataset is chosen
def get_coulomb_dimension():
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

def count_equal_elements(ges):
    """
    Utility function. Lets us count the number of inputs in binarized matrices
    that are the same, to make sure there's actually enough difference to 
    allow the NN to work
    
    Also counts the number of occurances of them
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

def example_chain(xyzFile):
    """
    Prints all steps of getting an input to a file, for demonstration purposes
    """
    f = open('example_input.txt', 'w')
    f2 = open(xyzFile, 'r')
    for line in f2.readlines():
        f.write(line)
    f2.close()
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=150)
    f.write('\n\n')
    c = generateCoulombMatrix(xyzFile)
    f.write(np.array2string(c))
    f.write('\n\n')
    rc = random_sort(c)
    f.write(np.array2string(rc))
    f.write('\n\n')
    bc = binarize_matrix(rc)
    f.write(np.array2string(bc))
    f.close()
    
def split_training():
    """
    writing for myself to split training data easily since we didn't
    run calculations for testing or validation sets
    """
    trainingWriter = open ('training_writer', 'w')
    testingWriter = open ('testing_writer', 'w')
    validationWriter = open ('validation_writer', 'w')
    
    energiesFile = open ('data/gdb_subset/training/energies.dat', 'r')
    for line in energiesFile.readlines():
        molId = line.split()[0]
        mol = ('data/gdb_subset/training/mol' + molId + '.xyz')
        rand = rd.random()
        if rand < .1:
            testingWriter.write(line)
            os.rename(mol, 'data/gdb_subset/testing/mol' + molId + '.xyz')
        elif rand < .2:
            validationWriter.write(line)
            os.rename(mol, 'data/gdb_subset/validation/mol' + molId + '.xyz')
        else:
            trainingWriter.write(line)
    trainingWriter.close()
    testingWriter.close()
    validationWriter.close()
    energiesFile.close()
            