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
        #To find the all the K nearest neighbors
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
        #count the class of the k nearest neighbors
        for i in range(k):
            if neighbors[i][1] == 2:
                count_2 +=1
            else:
                count_4 +=1
        #decide the class of the of the test point
        if count_2 > count_4:
            y_pred = np.append(y_pred,2)
        elif count_2 == count_4:
            y_pred = np.append(y_pred,neighbors[0][1])
        else:
            y_pred = np.append(y_pred,4)

    return y_pred
#End of Question 1

#*****************************************************************************
#Begin question 2 Evalutation[45%]
#*****************************************************************************
def Cross_Validation(data,k,p):
    #shuffle data
    np.random.shuffle(data)
    #calculate the fold_size
    fold_begin = 0
    fold_end = int(np.size(data,0)/10)
    fold_size = fold_end -fold_begin
    #measurements for the data
    error_rate = []
    accuracy = []
    sensitivity = []
    specificity = []
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

        #retrive the actual values of the testing points
        actual = data[fold_begin:fold_end,9]

        #Knn_Classifier to retrive predicted values
        pred = Knn_Classifier(x_test,x_train,y_train,k,p)

        #Side by side comparison between predicted and actual class values
        sidebyside = np.column_stack((actual,pred))
        #print(sidebyside)

        #Calculate the error, accuracy, sensitivity and specificity
        error = 0.0
        sens1 = 0.0
        sens2 = 0.0
        speci1 = 0.0
        speci2 = 0.0
        for i in range(len(actual)):
            if(actual[i]==2 ):          #sensitivity: total number of benign
                sens1 +=1
            elif(actual[i]==4 ):        #specificity: total number of malignant
                speci1 +=1
            if(pred[i] == 2 and actual[i] == 2):     #sensitivity: # predicted benign /  total number of benign
                sens2 +=1
            elif(pred[i] == 4 and actual[i] == 4):   #specificity: # predicted malignant / total number of malignant
                speci2 +=1

            if(actual[i] != pred[i] ):
                error+=1
        sensitivity.append(sens2/sens1)
        specificity.append(speci2/speci1)
        error_rate.append(float(error/fold_size))
        accuracy.append(1 - error_rate[j])
        # print("Error rate "+ str(error_rate[j]))
        # print("Accuracy  "+ str(accuracy[j]))
        # print("Sensitivity " + str(sensitivity[j]))
        # print("Specificity " + str(specificity[j]))

        fold_begin = fold_end
        fold_end = fold_end + fold_size
    return (error_rate,accuracy,sensitivity,specificity)
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
        #print("Error on training = " + str(total_error))
        # raw_input()
        if(total_error <= .036):
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

def Cross_Validation_perceptron(data,Random_weights):
    #shuffle data
    np.random.shuffle(data)
    #calculate the fold_size
    fold_begin = 0
    fold_end = int(np.size(data,0)/10)
    fold_size = fold_end

    #measurements for the data
    error_rate = []
    accuracy = []
    sensitivity = []
    specificity = []
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
        if(Random_weights):
            w_init = 2*np.random.rand(10)-1     #random values between [-1,1]
        else:
            w_init = np.zeros(10,dtype=float)   #0 Initialized weights

        #print(w_init)
        #raw_input()

        #  /-----------------------------------------------------
        #/train Perceptron retrive the wights for the Perceptron
        #\------------------------------------------------------
        weights = train_perceptron(input_x,output_y, w_init)

        #  /-----------------------------------------------------
        #/evaluate the Perceptron classifier ///below are some good weights
        #\------------------------------------------------------
        #weights = np.array([690.1052085,-53.76081527,-39.50214537,-19.1930896, -3.8676253,18.82719073,-21.11888355,6.07904847,-11.6877823,-43.66775949])
        #weights = np.array([68.89660005, -4.54788773, -4.04097265, -2.2572996, -0.30597967,  0.87090774,-2.26879302, 0.15780697, -0.09539544, -5.01148695])
        #weights = np.array([767.19134688, -48.61536643, -39.01605339, -20.12235216,  -9.39955249, 6.3184352, -26.41873175,  -2.28242008, -13.55489422, -47.20376838])
        pred = classy_perceptron(test_x, weights)

        #Side by side comparison between predicted and actual class valuesKnn_Classifier
        sidebyside = np.column_stack((actual,pred))
        #print(sidebyside)

        #Calculate the error, accuracy, sensitivity and specificity
        error = 0.0
        sens1 = 0.0
        sens2 = 0.0
        speci1 = 0.0
        speci2 = 0.0
        for i in range(len(actual)):
            if(actual[i]==2 ):          #sensitivity: total number of benign
                sens1 +=1
            elif(actual[i]==4 ):        #specificity: total number of malignant
                speci1 +=1
            if(pred[i] == 2 and actual[i] == 2):     #sensitivity: # predicted benign /  total number of benign
                sens2 +=1
            elif(pred[i] == 4 and actual[i] == 4):   #specificity: # predicted malignant / total number of malignant
                speci2 +=1

            if(actual[i] != pred[i] ):
                error+=1
        sensitivity.append(sens2/sens1)
        specificity.append(speci2/speci1)
        error_rate.append(float(error/fold_size))
        accuracy.append(1 - error_rate[j])
        # print("Error rate "+ str(error_rate[j]))
        # print("Accuracy  "+ str(accuracy[j]))
        # print("Sensitivity " + str(sensitivity[j]))
        # print("Specificity " + str(specificity[j]))

        fold_begin = fold_end
        fold_end = fold_end + fold_size

    return (error_rate,accuracy,sensitivity,specificity)


if __name__ == '__main__':
    data = Get_Data()

    #

    #************************************
    #Question 2 graphs
    #************************************
    '''
    knn_error = [None]*10
    knn_accuracy = [None]*10
    knn_sensitivity = [None]*10
    knn_specificity = [None]*10

    accuracy_std = []
    accuracy_mean = []

    sensitivity_std = []
    sensitivity_mean = []

    specificity_std = []
    specificity_mean = []
    j = 1
    for i in range (1,11):
        print("p = "+str(j)+" k = "+str(i))
        knn_error[i-1],knn_accuracy[i-1],knn_sensitivity[i-1],knn_specificity[i-1] = Cross_Validation(data,i,j)
        accuracy_std.append(np.std(np.array(knn_accuracy[i-1])))
        accuracy_mean.append(np.mean(np.array(knn_accuracy[i-1])))

        sensitivity_std.append(np.std(np.array(knn_sensitivity[i-1])))
        sensitivity_mean.append(np.mean(np.array(knn_sensitivity[i-1])))

        specificity_std.append(np.std(np.array(knn_specificity[i-1])))
        specificity_mean.append(np.mean(np.array(knn_specificity[i-1])))

    accuracy = plt.figure(1)
    x = np.arange(1, 11)
    y = accuracy_mean
    accuracy = plt.errorbar(x, y, xerr = 0, yerr=accuracy_std, color = 'green', ecolor='crimson',capsize=5, capthick=2  )
    plt.xlabel("K = number of nearest neighbors")
    plt.ylabel("Accuracy Performace")
    plt.title("Error_Bar Accuracy vs K with p =" + str(j))

    sensitivity = plt.figure(2)
    x = np.arange(1, 11)
    y = sensitivity_mean
    sensitivity = plt.errorbar(x, y, xerr = 0, yerr=sensitivity_std, color = 'indigo', ecolor='lightsalmon', capsize=5, capthick=2 )
    plt.xlabel("K = number of nearest neighbors")
    plt.ylabel("Sensitivity Performace")
    plt.title("Error_Bar Sensitivity vs K with p =" + str(j))

    specificity = plt.figure(3)
    x = np.arange(1, 11)
    y = specificity_mean
    specificity = plt.errorbar(x, y, xerr = 0, yerr=specificity_std, color = 'deepskyblue', ecolor='plum',capsize=5, capthick=2  )
    plt.xlabel("K = number of nearest neighbors")
    plt.ylabel("Specificity Performace")
    plt.title("Error_Bar Specificity vs K with p =" + str(j))

    plt.show()

    raw_input()
    '''
    #************************************
    #Question 3 graphs
    #************************************
    accuracy_std = []
    accuracy_mean = []

    sensitivity_std = []
    sensitivity_mean = []

    specificity_std = []
    specificity_mean = []

    for i in range (10):
        print("Iteration: "+str(i))

        p_error,p_accuracy,p_sensitivity,p_specificity = Cross_Validation_perceptron(data,Random_weights=True)

        accuracy_std.append(np.std(np.array(p_accuracy)))
        accuracy_mean.append(np.mean(np.array(p_accuracy)))

        sensitivity_std.append(np.std(np.array(p_sensitivity)))
        sensitivity_mean.append(np.mean(np.array(p_sensitivity)))

        specificity_std.append(np.std(np.array(p_specificity)))
        specificity_mean.append(np.mean(np.array(p_specificity)))

    accuracy = plt.figure(1)
    x = np.arange(1, 11)
    y = accuracy_mean
    accuracy = plt.errorbar(x, y, xerr = 0, yerr=accuracy_std, color = 'y', ecolor='indianred',capsize=5, capthick=2  )
    plt.xlabel("10 independent instances")
    plt.ylabel("Accuracy Performace")
    plt.title("Error_Bar Accuracy vs Instance with random init weights" )

    sensitivity = plt.figure(2)
    x = np.arange(1, 11)
    y = sensitivity_mean
    sensitivity = plt.errorbar(x, y, xerr = 0, yerr=sensitivity_std, color = 'blueviolet', ecolor='lightpink', capsize=5, capthick=2 )
    plt.xlabel("10 independent instances")
    plt.ylabel("Sensitivity Performace")
    plt.title("Error_Bar Sensitivity vs Instance with random init weights")

    specificity = plt.figure(3)
    x = np.arange(1, 11)
    y = specificity_mean
    specificity = plt.errorbar(x, y, xerr = 0, yerr=specificity_std, color = 'royalblue', ecolor='violet',capsize=5, capthick=2  )
    plt.xlabel("10 independent instances")
    plt.ylabel("Specificity Performace")
    plt.title("Error_Bar Specificity vs Instance with random init weights" )

    plt.show()

    raw_input()
