#Lino Valdovinos
#Cs-171 Machine Learingin and Data mining
#Assignment 2
#
import os
import numpy as np
import matplotlib.pyplot as plt


#Questions 0 Getting raw data[ 5%]
#Working with
def Get_Data():
    fname = 'breast-cancer-wisconsin.data'
    data = np.loadtxt(fname, delimiter=',') #Removed instances of missing data
    data = data.reshape(683,11)
    data = data[:,(1,2,3,4,5,6,7,8,9,10)]
    return data

#Question 1: K-Nearest neighbor classifier[50%]
def distance(x,y,p):
    distance = 0;
    for i in range(len(x)):
        distance += (abs(x[i] -y[i]))**p
    return((distance)**(1.0/p))

def getKey(item):
    return item[0]
#x_test data ; x_train data matrix of features; y_train data vector of lables for x_train
#k = the number of nearest neighbors; p = Lp distance value
#y_pred vector of predicted lables for x_test
def Knn_Classifier(x_test,x_train,y_train,k,p):

    #Initialized a numpy array for y_pred
    y_pred = np.array([])
    #outer loop iterates through x_test points
    for x in range(np.size(x_test,0)):
        d_old = 1000000.0;
        k_count = 0
        neighbors = []
        #Inner for loop iterates through training points
        for y in range(np.size(y_train)):
            d_new = distance(x_test[x],x_train[y],p)
            if d_new < d_old or k_count <= k :
                k_count+=1
                d_old = d_new
                neighbors.append([d_new,y_train[y]])
        #sort the nearest neighbors by distance
        neighbors = sorted(neighbors,key=getKey)
        count_2 = 0
        count_4 = 0
        #print(neighbors)
        #print("************************************")
        #count the class of the k nearest neighbors
        for i in range(k):
            if neighbors[i][1] == 2:
                count_2 +=1
            else:
                count_4 +=1
        #decide the class of the of the test point
        if count_2 > count_4:
            y_pred = np.append(y_pred,2)
        else:
            y_pred = np.append(y_pred,4)

    return y_pred
#End of Question 1

#Begin question 2 Evalutation[45%]
def Cross_Validation(data,k,p):
    #calculate the fold_size
    fold_begin = 0
    fold_end = int(np.size(data,0)/10)
    fold_size = fold_end
    error_rate = []
    #iterates through the data 10 times
    for j in range(10):
        #append extra points to the last iteration
        if j == 9:
            fold_end = np.size(data,0)
            fold_size = fold_end - fold_begin
        #Calculate the test and training data
        x_test = data[fold_begin:fold_end,(1,2,3,4,5,6,7,8)]
        x_train = data[:,(1,2,3,4,5,6,7,8)]
        y_train = data[:,9]
        remove_test_from_train = np.arange(fold_begin,fold_end)
        x_train = np.delete(x_train,remove_test_from_train,0)
        y_train = np.delete(y_train,remove_test_from_train,0)
        print("Fold range :" +str(fold_begin)+" - " + str(fold_end))

        #retrive the actual values
        actual = data[fold_begin:fold_end,9]

        #Knn_Classifier to retrive predicted values
        pred = Knn_Classifier(x_test,x_train,y_train,k,p)

        #Side by side comparison between predicted and actual class values
        sidebyside = np.hstack((actual,pred))
        sidebyside = sidebyside.reshape((fold_size,2))
        print(sidebyside)

        #Calculate the error
        error = 0.0
        for i in range(len(actual)):
            if(actual[i] != pred[i] ):
                error+=1
        error_rate.append(float(error/fold_size))
        print("Error rate "+ str(error_rate[j]))

        raw_input()
        fold_begin = fold_end
        fold_end = fold_end + fold_size
    return (error_rate)
#END of question 2 Evalutation

#Question 3 Perceptron [30%]-Extra Credit

#activation function
def sine(a):
    if a > 0.0:
        return(2)
    else:
        return(4)
#w_init => is the Initialization for the weights
def train_perceptron(input_x, output_y, w_init):
    pass


def classy_perceptron(input_x,w):
    pred_sum = 0.0;
    pred = [];
    for i in range(np.size(input_x,0)):
        for j in range(np.size(w)):
            pred_sum += input_x[i][j] * w[j]
        pred.append(sine(pred_sum))
    return pred

def Cross_Validation_perceptron(data,k,p):
    #calculate the fold_size
    fold_begin = 0
    fold_end = int(np.size(data,0)/10)
    fold_size = fold_end

    error_rate = [] #Initialized list that is returned

    #iterates through the data 10 times
    for j in range(10):
        #append extra instance to the last iteration
        if j == 9:
            fold_end = np.size(data,0)
            fold_size = fold_end - fold_begin

        #Calculate the testing and training data for this fold
        test_x = data[fold_begin:fold_end,(1,2,3,4,5,6,7,8)]
        input_x = data[:,(1,2,3,4,5,6,7,8)]
        output_y = data[:,9]
        remove_test_from_train = np.arange(fold_begin,fold_end)
        input_x = np.delete(input_x,remove_test_from_train,0)
        output_y = np.delete(output_y,remove_test_from_train,0)
        print("Fold range :" +str(fold_begin)+" - " + str(fold_end))

        #retrive the actual values
        actual = data[fold_begin:fold_end,9]
        #Initialized weights
        w_init = np.zeros(fold_size)

        #  /-----------------------------------------------------
        #/train Perceptron retrive the wights for the Perceptron
        #\------------------------------------------------------
        weights = train_perceptron(input_x,output_y, w_init)

        #  /-----------------------------------------------------
        #/evaluate the Perceptron classifier
        #\------------------------------------------------------
        pred = classify_perceptron(test_x, weights)

        #Side by side comparison between predicted and actual class valuesKnn_Classifier
        sidebyside = np.hstack((actual,pred))
        sidebyside = sidebyside.reshape((fold_size,2))
        print(sidebyside)

        #Calculate the error
        error = 0.0
        for i in range(len(actual)):
            if(actual[i] != pred[i] ):
                error+=1
        error_rate.append(float(error/fold_size))
        print("Error rate "+ str(error_rate[j]))

        raw_input()
        fold_begin = fold_end
        fold_end = fold_end + fold_size

    return (error_rate)


if __name__ == '__main__':
    data = Get_Data()
    knn_error = Cross_Validation(data,3,2)
