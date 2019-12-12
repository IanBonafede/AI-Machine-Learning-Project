import numpy as np
import random
import struct
import os

def readFile(filename):
    filehandle = open(filename)
    print(filehandle.read())
    filehandle.close()

def loadsparsedata(fn):
    
    fp = open(fn)
    lines = fp.readlines()

    numFeatures = 11

    data = np.zeros((len(lines),numFeatures))
    y = 0
    x = 0
    for line in lines:
        x=0
        for i in line.split():
            data[y][x] = float(i)
            #print(data[y][x])
            x+=1
        y+=1
    
    
    
    return data

def leave_one_out_cross_validation(data,current_set,feature_to_add):
    return random.random()        # This is a testing stub only



def  feature_search_demo(data):
 
    current_set_of_features = np.array(()) #Initialize an empty set
 
    for i in range(1, data.shape[1]):
        print('On the',i,'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0;    
    
        for k in range(1, data.shape[1]):
            #print(np.intersect1d(current_set_of_features, k))
            if (np.intersect1d(current_set_of_features, k)).size == 0: # Only consider adding, if not already added.
                print('--Considering adding the ', k,' feature')
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k+1)
        
                if accuracy > best_so_far_accuracy: 
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        print('On level', i,',added feature', feature_to_add_at_this_level, 'to current set')
        current_set_of_features = np.append(current_set_of_features, [feature_to_add_at_this_level])
        print('On level', i,'current set of features ', current_set_of_features)

    print('set of features ', current_set_of_features)
        



#For accessing the file in the same folder
filename = "/home/ian/Documents/CS_170/project_2/CS170_SMALLtestdata/CS170_SMALLtestdata__80.txt"
print(filename)
data = loadsparsedata(filename)
feature_search_demo(data)

