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
#x_test data ; x_train data matrix of features; y_train data vector of lables for x_train
#k = the number of nearest neighbors; p = Lp distance value
#y_pred vector of predicted lables for x_test
def getKey(item):
    return item[0]

def Knn_Classifier(x_test,x_train,y_train,k,p):

    y_pred = np.array([])

    k_count = 0
    for x in range(np.size(x_test,0)):
        d_old = 1000000.0;
        neighbors = []
        for y in range(np.size(y_train)):
            d_new = distance(x_test[x],x_train[y],p)
            if d_new < d_old or k_count <= k:
                k_count+=1
                d_old = d_new
                neighbors.append([d_new,y_train[y]])
        #print(neighbors)
        neighbors = sorted(neighbors,key=getKey)
        #print(neighbors)

        count_2 = 0
        count_4 = 0
        for i in range(k):
            if neighbors[i][1] == 2:
                count_2 +=1
            else:
                count_4 +=1
        if count_2 > count_4:
            y_pred = np.append(y_pred,2)
        else:
            y_pred = np.append(y_pred,4)

    return y_pred
#End of Question 1

#Begin question 2 Evalutation[45%]
def Cross_Validation(data,k,p):
    fold_begin = 0
    fold_end = np.size(data,0)
    fold_end = int(fold_end/10)
    fold_size = fold_end
    for j in range(10):
        print("Fold range :" +str(fold_begin)+" - " + str(fold_end))
        x_test = data[fold_begin:fold_end,(1,2,3,4,5,6,7,8)]
        x_train = data[fold_end:682,(1,2,3,4,5,6,7,8)]
        y_train = data[fold_end:682,9]
        actual = data[fold_begin:fold_end,9]
        pred = Knn_Classifier(x_test,x_train,y_train,k,p)
        sidebyside = np.hstack((actual,pred))
        sidebyside = sidebyside.reshape((fold_size,2))
        print(sidebyside)
        error = 0.0
        for i in range(len(actual)):
            if(actual[i] != pred[i] ):
                error+=1

        print("Error rate "+ str(float(error/fold_size)))

        raw_input()
        fold_begin = fold_end
        fold_end = fold_end + fold_size


if __name__ == '__main__':
    data = Get_Data()
    Cross_Validation(data,3,2)
