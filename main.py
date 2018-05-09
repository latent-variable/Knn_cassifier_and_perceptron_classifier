#Lino Valdovinos
#Cs-171 Machine Learingin and Data mining
#Assignment 2
#
import os
import numpy as np
import matplotlib.pyplot as plt

#*****************************************************************************
#Questions 0 Getting raw data[ 5%]
#*****************************************************************************
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

#*****************************************************************************
#Begin question 2 Evalutation[45%]
#*****************************************************************************
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
        x_test = data[fold_begin:fold_end,(0,1,2,3,4,5,6,7,8)]
        x_train = data[:,(0,1,2,3,4,5,6,7,8)]
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
        sidebyside = np.column_stack((actual,pred))
        #print(sidebyside)

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
#*****************************************************************************
#Question 3 Perceptron [30%]-Extra Credit
#*****************************************************************************
#activation function
def sign(a):
    if a > 0.0:
        return(1.0)   #class 2
    else:
        return(-1.0)  #class 4

#w => is the Initialization for the weights and input_x is training data
def train_perceptron(input_x, output_y, w):
    wrong = True
    while wrong:     #iterate untill error rate is below 2.6%
        total_error = 0.0
        for i in range(np.size(input_x,0)):

            pred_sum = 0.0;
            for j in range(np.size(w)):
                pred_sum += input_x[i][j] * w[j]

            #convert outputs from 2 ->1 and 4 -> -1
            if output_y[i] == 2:
                d = 1.0
            elif output_y[i] == 4:
                d = -1

            #calculate error can be [0,2,-2]
            error = d - sign(pred_sum)
            if error != 0.0:
                total_error += 1.0
                #Correct the weights
                for k in range(np.size(w)):
                    w[k] += error*input_x[i][k]

        #calculate error, if needed keep training
        total_error = total_error/float(np.size(input_x,0))
        print("Error on training = " + str(total_error))
        # raw_input()
        if(total_error <= .027):
            print (w)
            wrong = False

    return w
#Once perceptron is trained,classifier read
def classy_perceptron(input_x,w):
    pred = [];
    for i in range(np.size(input_x,0)):
        pred_sum = 0.0;
        #sum of features*weights
        for j in range(np.size(w)):
            pred_sum += input_x[i][j] * w[j]
        #activation function
        if sign(pred_sum) == 1:
            pred.append(2)
        elif sign(pred_sum) == -1:
            pred.append(4)
    return pred

def Cross_Validation_perceptron(data):
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
        test_x = data[fold_begin:fold_end,(0,1,2,3,4,5,6,7,8)]
        input_x = data[:,(0,1,2,3,4,5,6,7,8)]
        output_y = data[:,9]
        remove_test_from_train = np.arange(fold_begin,fold_end)
        input_x = np.delete(input_x,remove_test_from_train,0)
        output_y = np.delete(output_y,remove_test_from_train,0)

        x0_input = np.ones(np.size(output_y)).reshape((np.size(output_y),1))
        x0_test = np.ones(np.size(test_x,0)).reshape((np.size(test_x,0),1))
        input_x = np.column_stack((x0_input,input_x))

        test_x = np.column_stack((x0_test,test_x))

        print("Fold range :" +str(fold_begin)+" - " + str(fold_end))

        #retrive the actual values
        actual = data[fold_begin:fold_end,9]
        #Initialized weights
        w_init = np.zeros(10,dtype=float)
        #w_init = 2*np.random.rand(10)-1

        print(w_init)
        raw_input()

        #  /-----------------------------------------------------
        #/train Perceptron retrive the wights for the Perceptron
        #\------------------------------------------------------
        weights = train_perceptron(input_x,output_y, w_init)

        #  /-----------------------------------------------------
        #/evaluate the Perceptron classifier
        #\------------------------------------------------------
        #weights = np.array([690.1052085,-53.76081527,-39.50214537,-19.1930896, -3.8676253,18.82719073,-21.11888355,6.07904847,-11.6877823,-43.66775949])
        #weights = np.array([68.89660005, -4.54788773, -4.04097265, -2.2572996, -0.30597967,  0.87090774,-2.26879302, 0.15780697, -0.09539544, -5.01148695])
        #weights = np.array([767.19134688, -48.61536643, -39.01605339, -20.12235216,  -9.39955249, 6.3184352, -26.41873175,  -2.28242008, -13.55489422, -47.20376838])
        pred = classy_perceptron(test_x, weights)

        #Side by side comparison between predicted and actual class valuesKnn_Classifier
        sidebyside = np.column_stack((actual,pred))
        #print(sidebyside)

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
    #knn_error = Cross_Validation(data,3,2)
    Perceptron_error = Cross_Validation_perceptron(data)
