import numpy as np
import random
import math
import time
import struct
import os

def readFile(filename):
    filehandle = open(filename)
    print(filehandle.read())
    filehandle.close()

def loadsparsedata(fn):
    fp1 = open(fn)
    firstLine = fp1.readline()
    numColumns = 0
    for i in firstLine.split():
        numColumns+=1

    fp = open(fn)
    lines = fp.readlines()
    
    
    

    data = np.zeros((len(lines),numColumns))
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


def leave_one_out_add(data,current_set,feature_to_add):
    num_correct = 0
    features = np.append(current_set, [feature_to_add])
    # print(features)
 
    for i in range(data.shape[0]):
        best_so_far = float("inf")
        best_so_far_loc = -1
        for j in  range(data.shape[0]):
            if i != j: 
                sum_of_squares = 0
                for  feature in features:
                    sum_of_squares += math.pow(data[i][feature] -data[j][feature], 2)
                distance = math.sqrt(sum_of_squares)
                if distance < best_so_far:
                    best_so_far = distance
                    best_so_far_loc = j

        if data[i][0] == data[best_so_far_loc][0]:
            # print(' * exemplar', i, 'correct')
            num_correct += 1
    # print(' * accuracy of adding', feature_to_add, ':', num_correct/data.shape[0])
    return num_correct/data.shape[0]

def leave_one_out_remove(data,current_set,feature_to_remove):
    num_correct = 0
    if feature_to_remove == -1:
        features = current_set
    else:
        features = np.delete(current_set, feature_to_remove)
    # print(features)
 
    for i in range(data.shape[0]):
        best_so_far = float("inf")
        best_so_far_loc = -1
        for j in  range(data.shape[0]):
            if i != j: 
                sum_of_squares = 0
                for  feature in features:
                    sum_of_squares += math.pow(data[i][feature] -data[j][feature], 2)
                distance = math.sqrt(sum_of_squares)
                if distance < best_so_far:
                    best_so_far = distance
                    best_so_far_loc = j

        if data[i][0] == data[best_so_far_loc][0]:
            # print(' * exemplar', i, 'correct')
            num_correct += 1
    # print(' * accuracy of removing', current_set[feature_to_remove], ':', num_correct/data.shape[0])
    return num_correct/data.shape[0]


def  forward_selection(data):
    start = time.time()
 
    current_set_of_features = np.array(()) #Initialize an empty set
    current_set_of_features = current_set_of_features.astype(int)

    best_overall_combination = []
    best_overall_accuracy = 0
 
    for i in range(1, data.shape[1]):
        #print('On the',i,'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0;    
    
        for j in range(1, data.shape[1]):
            #print(np.intersect1d(current_set_of_features, k))
            if (np.intersect1d(current_set_of_features, j)).size == 0: # Only consider adding, if not already added.
                #print('--Considering adding the', j, 'feature')
                #accuracy = leave_one_out_cross_validation(data,current_set_of_features,j)
                accuracy = leave_one_out_add(data,current_set_of_features,j)

                if accuracy > best_so_far_accuracy: 
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j

        #print('On level', i,',added feature', feature_to_add_at_this_level, 'to current set')
        

        current_set_of_features = np.append(current_set_of_features, [ feature_to_add_at_this_level])
        print('Level', i,'current set of features ', current_set_of_features)
        print('Accuracy:', best_so_far_accuracy)

        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_overall_combination = current_set_of_features

    print('Best set of features', best_overall_combination)
    print('Accuracy:', best_overall_accuracy)
    end = time.time()
    print('time (s):', end - start)


def  backwards_elimination(data):
    start = time.time()
    
    best_overall_combination = []
    best_overall_accuracy = 0

    current_set_of_features = np.arange(1, data.shape[1]) #Initialize an empty set
    current_set_of_features = current_set_of_features.astype(int)
    best_so_far_accuracy = leave_one_out_remove(data,current_set_of_features,-1)
    print('Level 0 current set of features ', current_set_of_features)
    print('Accuracy:', best_so_far_accuracy)

 
    for i in range(1, data.shape[1] - 1 ):
        # print('On the',i,'th level of the search tree')
        accuracy = leave_one_out_remove(data,current_set_of_features, -1)
        feature_to_remove_at_this_level = []
        best_so_far_accuracy = 0;    
    
        for j in range(current_set_of_features.shape[0]):
            #print(np.intersect1d(current_set_of_features, k))
                # print('--Considering removing the', current_set_of_features[j],'feature')
                #accuracy = leave_one_out_cross_validation(data,current_set_of_features,j)
                accuracy = leave_one_out_remove(data,current_set_of_features,j)

                if accuracy > best_so_far_accuracy: 
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = j

        # print('On level', i,',removed feature', current_set_of_features[feature_to_remove_at_this_level], 'from current set')
        current_set_of_features = np.delete(current_set_of_features, feature_to_remove_at_this_level)
        print('Level', i,'current set of features ', current_set_of_features)
        print('Accuracy:', best_so_far_accuracy)

        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_overall_combination = current_set_of_features

    print('Best set of features', best_overall_combination)
    print('Accuracy:', best_overall_accuracy)
    end = time.time()
    print('time (s):', end - start)


def  original_search(data):
    start = time.time()
 
    current_set_of_features = np.array(()) #Initialize an empty set
    current_set_of_features = current_set_of_features.astype(int)


    best_overall_combination = []
    best_overall_accuracy = 0
    last_accuracy = 1

 
    for i in range(1, data.shape[1]):
        #print('On the',i,'th level of the search tree')
        feature_to_add_at_this_level = []
        best_so_far_accuracy = 0  
    
        for j in range(1, data.shape[1]):
            #print(np.intersect1d(current_set_of_features, k))
            if (np.intersect1d(current_set_of_features, j)).size == 0: # Only consider adding, if not already added.
                #print('--Considering adding the', j, 'feature')
                #accuracy = leave_one_out_cross_validation(data,current_set_of_features,j)
                accuracy = leave_one_out_add(data,current_set_of_features,j)

                if accuracy > best_so_far_accuracy: 
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = j

        #print('On level', i,',added feature', feature_to_add_at_this_level, 'to current set')
        current_set_of_features = np.append(current_set_of_features, [ feature_to_add_at_this_level])
        print('Level', i,'current set of features ', current_set_of_features)
        print('Accuracy:', best_so_far_accuracy)

        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_overall_combination = current_set_of_features
        elif best_so_far_accuracy < best_overall_accuracy:
            break

    
    print('Best set of features', best_overall_combination)
    print('Accuracy:', best_overall_accuracy)
    end = time.time()
    print('time (s):', end - start)




smallFile = "/home/ian/Documents/CS_170/project_2/CS170_SMALLtestdata/CS170_SMALLtestdata__88.txt"
print(smallFile)
smallData = loadsparsedata(smallFile)
print('-----Forward Selection-----')
forward_selection(smallData)
print('---Backwards Elimination---')
backwards_elimination(smallData)
print('------original Search------')
original_search(smallData)

largeFile = "/home/ian/Documents/CS_170/project_2/CS170_LARGEtestdata/CS170_LARGEtestdata__90.txt"
print(largeFile)
largeData = loadsparsedata(largeFile)
print('-----Forward Selection-----')
forward_selection(largeData)
print('---Backwards Elimination---')
backwards_elimination(largeData)
print('------original Search------')
original_search(largeData)
