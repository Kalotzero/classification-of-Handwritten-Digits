import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

#Uploading the trainning and test data

training_data=pd.read_csv('C:\\Users\\Sakis\\Desktop\\ΜΑΠ\\DATA ANALYSIS\\3η Εργασία\\project_svd_handwritten-3\\training_data.csv',sep=';')
test_data=pd.read_csv('C:\\Users\\Sakis\\Desktop\\ΜΑΠ\\DATA ANALYSIS\\3η Εργασία\\project_svd_handwritten-3\\test_data.csv',sep=';')
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]
class6=[]
class7=[]
class8=[]
class9=[]
class0=[]

#converting the data into float in order to be the calculation more easy
training_data=training_data.astype(float)
test_data=test_data.astype(float)



#Using a loop to classify yhe trainning(known) data in to 10 different classes
#each class represents a number for example class2 the number 2 , class 0 the number 0

for i in training_data:
    if (int(float(training_data[i].name)))==1:
        class1.append(training_data[i])
    elif (int(float(training_data[i].name)))==2:
        class2.append(training_data[i])
    elif (int(float(training_data[i].name)))==3:
        class3.append(training_data[i])
    elif (int(float(training_data[i].name)))==4:
        class4.append(training_data[i])
    elif (int(float(training_data[i].name)))==5:
        class5.append(training_data[i])
    elif (int(float(training_data[i].name)))==6:
        class6.append(training_data[i])
    elif (int(float(training_data[i].name)))==7:
        class7.append(training_data[i])
    elif (int(float(training_data[i].name)))==8:
        class8.append(training_data[i])
    elif (int(float(training_data[i].name)))==9:
        class9.append(training_data[i])
    else:
        class0.append(training_data[i])


#converting the matrixes to be more easy to analyze the data

class1=pd.DataFrame(class1).transpose()
class2=pd.DataFrame(class2).transpose()
class3=pd.DataFrame(class3).transpose()
class4=pd.DataFrame(class4).transpose()
class5=pd.DataFrame(class5).transpose()
class6=pd.DataFrame(class6).transpose()
class7=pd.DataFrame(class7).transpose()
class8=pd.DataFrame(class8).transpose()
class9=pd.DataFrame(class9).transpose()
class0=pd.DataFrame(class0).transpose()



#the below f function is created to perfom the SVD function in x matrix and k represents the singular values tha we want to use

def f(x,k):
    U,S,V=linalg.svd(x,full_matrices=False)
    n=np.matrix(U[:,:k])@np.diag(S[:k])@np.matrix(V[:k,:])
    return n



#calling the f function and using k=20(singular values) for all thw claasses

n_1=pd.DataFrame(f(class1,20))
n_2=pd.DataFrame(f(class2,20))
n_3=pd.DataFrame(f(class3,20))
n_4=pd.DataFrame(f(class4,40))
n_5=pd.DataFrame(f(class5,20))
n_6=pd.DataFrame(f(class6,20))
n_7=pd.DataFrame(f(class7,20))
n_8=pd.DataFrame(f(class8,20))
n_9=pd.DataFrame(f(class9,20))
n_0=pd.DataFrame(f(class0,20))




#Below we use a double loop. FIrst we take the first number from the test(unknown) data which is the  vector of ith column of test data
#In the second loop we try to find which of the 20 singular values(of each class of number) is more near to the ith vector of test data. We use the norm 2 to examinate this
#After we keep the smaller difference and we append the 10 smaller norms(the smaller of each number 0-9) in a new list.
#last we try with if to test and find in which number this norm belong in order to be classified
  
number_ones=[]
number_twos=[]
number_threes=[]
number_fours=[]
number_fives=[]
number_sixs=[]
number_sevens=[]
number_eights=[]
number_nines=[]
number_zeros=[]


for i in range(0,2006):
    for j in range(0,20):
        norm1=[]
        norm2=[]
        norm3=[]
        norm4=[]
        norm5=[]
        norm6=[]
        norm7=[]
        norm8=[]
        norm9=[]
        norm0=[]
        t=test_data.iloc[:,i]
        n1=n_1.iloc[:,j]
        n2=n_2.iloc[:,j]
        n3=n_3.iloc[:,j]
        n4=n_4.iloc[:,j]
        n5=n_5.iloc[:,j]
        n6=n_6.iloc[:,j]
        n7=n_7.iloc[:,j]
        n8=n_8.iloc[:,j]
        n9=n_9.iloc[:,j]
        n0=n_0.iloc[:,j]
        
        
        norm1.append(np.linalg.norm(t-n1))
        norm2.append(np.linalg.norm(t-n2))
        norm3.append(np.linalg.norm(t-n3))
        norm4.append(np.linalg.norm(t-n4))
        norm5.append(np.linalg.norm(t-n5))
        norm6.append(np.linalg.norm(t-n6))
        norm7.append(np.linalg.norm(t-n7))
        norm8.append(np.linalg.norm(t-n8))
        norm9.append(np.linalg.norm(t-n9))
        norm0.append(np.linalg.norm(t-n0))
        
        
        
    norm_1=min(norm1)
    norm_2=min(norm2)
    norm_3=min(norm3)
    norm_4=min(norm4)
    norm_5=min(norm5)
    norm_6=min(norm6)
    norm_7=min(norm7)
    norm_8=min(norm8)
    norm_9=min(norm9)
    norm_0=min(norm0)
    
    
    norms=[]
    norms.append(norm_1)
    norms.append(norm_2)
    norms.append(norm_3)
    norms.append(norm_4)
    norms.append(norm_5)
    norms.append(norm_6)
    norms.append(norm_7)
    norms.append(norm_8)
    norms.append(norm_9)
    norms.append(norm_0)
        
    min_norm=min(norms)


    if min_norm==norm_1:
        number_ones.append(test_data.iloc[:,i])
    elif min_norm==norm_2:
        number_twos.append(test_data.iloc[:,i])
    elif min_norm==norm_3:
        number_threes.append(test_data.iloc[:,i])
    elif min_norm==norm_4:
        number_fours.append(test_data.iloc[:,i])  
    elif min_norm==norm_5:
        number_fives.append(test_data.iloc[:,i])
    elif min_norm==norm_6:
        number_sixs.append(test_data.iloc[:,i])
    elif min_norm==norm_7:
        number_sevens.append(test_data.iloc[:,i])
    elif min_norm==norm_8:
        number_eights.append(test_data.iloc[:,i])
    elif min_norm==norm_9:
        number_nines.append(test_data.iloc[:,i])
    else:
        number_zeros.append(test_data.iloc[:,i])





#As for the trainning data we classify the test data in order to test the results and take a percentage of right classification

test_1=[]
test_2=[]
test_3=[]
test_4=[]
test_5=[]
test_6=[]
test_7=[]
test_8=[]
test_9=[]
test_0=[]



for i in test_data:
    if (int(float(test_data[i].name)))==1:
        test_1.append(test_data[i])
    elif (int(float(test_data[i].name)))==2:
        test_2.append(test_data[i])
    elif (int(float(test_data[i].name)))==3:
        test_3.append(test_data[i])
    elif (int(float(test_data[i].name)))==4:
        test_4.append(test_data[i])
    elif (int(float(test_data[i].name)))==5:
        test_5.append(test_data[i])
    elif (int(float(test_data[i].name)))==6:
        test_6.append(test_data[i])
    elif (int(float(test_data[i].name)))==7:
        test_7.append(test_data[i])
    elif (int(float(test_data[i].name)))==8:
        test_8.append(test_data[i])
    elif (int(float(test_data[i].name)))==9:
        test_9.append(test_data[i])
    else:
        test_0.append(test_data[i])




test_1=pd.DataFrame(test_1).transpose()
test_2=pd.DataFrame(test_2).transpose()
test_3=pd.DataFrame(test_3).transpose()
test_4=pd.DataFrame(test_4).transpose()
test_5=pd.DataFrame(test_5).transpose()
test_6=pd.DataFrame(test_6).transpose()
test_7=pd.DataFrame(test_7).transpose()
test_8=pd.DataFrame(test_8).transpose()
test_9=pd.DataFrame(test_9).transpose()
test_0=pd.DataFrame(test_0).transpose()




#In classified numbers we are putting the length of each class in which the numbers has been classified. First the number 0 and last the number 9 
#We are doing the same at test_numbers, now we are taking the test data

classified_numbers=[len(number_zeros),len(number_ones),len(number_twos),len(number_threes),len(number_fours),len(number_fives),len(number_sixs),len(number_sevens),len(number_eights),len(number_nines)]
test_numbers=[len(test_0.columns),len(test_1.columns),len(test_2.columns),len(test_3.columns),len(test_4.columns),len(test_5.columns),len(test_6.columns),len(test_7.columns),len(test_8.columns),len(test_9.columns)]

#using a loop to extract the percentage of success for all the numbers
percentage_of_numbers=[]

for i in range(10):
    percentage_of_numbers.append((classified_numbers[i]/test_numbers[i])*100)
    
#Putting the results into a matrix in order to observe them more carefully 
percentage_of_success=pd.DataFrame(data=(classified_numbers,test_numbers,percentage_of_numbers))

percentage_of_success=percentage_of_success.transpose()
percentage_of_success.columns=['classified_numbers','test_numbers','percentage_of_numbers']
print(percentage_of_success)


#Αfter our test we can see that the percentage of each number is : number 0 53.2%,number 1 159.42%,number 3 83.3%,number 3 103.01%
#number 4 19.5%,number 5 81.25%,number 6 134.7%,number 7 45.5%,number 8 100%,number 9 241.24%
#As we can observe the most attractive number is 9 and the less the number 4. Also number 0 and 7 have lowe percentage of succes

#Using the below code we can examinate each number if they are bad written
#k0 below refers to plotting the number 0 , we can use the same code for each number and use n_3 for number 3 instead of n_o for number 0
k0=np.matrix(n_0.iloc[:,1])
k0=k0.reshape(16,16)
plt.figure(figsize=(6,6))
plt.imshow(k0,aspect='auto')
plt.show()



#after a few personal experiments i found which singular values are better written
#So we are going to tune the algorithm with these values and after to examinate the results


number_ones=[]
number_twos=[]
number_threes=[]
number_fours=[]
number_fives=[]
number_sixs=[]
number_sevens=[]
number_eights=[]
number_nines=[]
number_zeros=[]


for i in range(0,2006):
    norms=[]
    t=test_data.iloc[:,i]
    n1=n_1.iloc[:,0]
    n2=n_2.iloc[:,1]
    n3=n_3.iloc[:,1]
    n4=n_4.iloc[:,6]
    n5=n_5.iloc[:,2]
    n6=n_6.iloc[:,2]
    n7=n_7.iloc[:,7]
    n8=n_8.iloc[:,8]
    n9=n_9.iloc[:,9]
    n0=n_0.iloc[:,6]
    norm1=np.linalg.norm(t-n1)
    norms.append(norm1)
    norm2=np.linalg.norm(t-n2)
    norms.append(norm2)
    norm3=np.linalg.norm(t-n3)
    norms.append(norm3)
    norm4=np.linalg.norm(t-n4)
    norms.append(norm4)
    norm5=np.linalg.norm(t-n5)
    norms.append(norm5)
    norm6=np.linalg.norm(t-n6)
    norms.append(norm6)
    norm7=np.linalg.norm(t-n7)
    norms.append(norm7)
    norm8=np.linalg.norm(t-n8)
    norms.append(norm8)
    norm9=np.linalg.norm(t-n9)
    norms.append(norm9)
    norm0=np.linalg.norm(t-n0)
    norms.append(norm0)
    min_norm=min(norms)
    
    
    if min_norm==norm1:
        number_ones.append(test_data.iloc[:,i])
    elif min_norm==norm2:
        number_twos.append(test_data.iloc[:,i])
    elif min_norm==norm3:
        number_threes.append(test_data.iloc[:,i])
    elif min_norm==norm4:
        number_fours.append(test_data.iloc[:,i])  
    elif min_norm==norm5:
        number_fives.append(test_data.iloc[:,i])
    elif min_norm==norm6:
        number_sixs.append(test_data.iloc[:,i])
    elif min_norm==norm7:
        number_sevens.append(test_data.iloc[:,i])
    elif min_norm==norm8:
        number_eights.append(test_data.iloc[:,i])
    elif min_norm==norm9:
        number_nines.append(test_data.iloc[:,i])
    else:
        number_zeros.append(test_data.iloc[:,i])
        
#We are doing exactly the same as the fist time
classified_numbers=[len(number_zeros),len(number_ones),len(number_twos),len(number_threes),len(number_fours),len(number_fives),len(number_sixs),len(number_sevens),len(number_eights),len(number_nines)]
test_numbers=[len(test_0.columns),len(test_1.columns),len(test_2.columns),len(test_3.columns),len(test_4.columns),len(test_5.columns),len(test_6.columns),len(test_7.columns),len(test_8.columns),len(test_9.columns)]

#using a loop to extract the percentage of success for all the numbers
percentage_of_numbers=[]

for i in range(10):
    percentage_of_numbers.append((classified_numbers[i]/test_numbers[i])*100)
    
#Putting the results into a matrix in order to observe them more carefully 
percentage_of_success=pd.DataFrame(data=(classified_numbers,test_numbers,percentage_of_numbers))

percentage_of_success=percentage_of_success.transpose()
percentage_of_success.columns=['classified_numbers','test_numbers','percentage_of_numbers']
print(percentage_of_success)
        


#Αfter the second test we can see that the percentage of each number is : number 0 53.2%,number 1 127.27%,number 2 122.2%,number 3 115.01%
#number 4 40.5%,number 5 141.87%,number 6 106.4%,number 7 106.1%,number 8 81.3%,number 9 150%
#Generally speaking we can say that the new algorithm  balanced a little bit the results
#I observed that the number 5 is the most confusing number to classify
#when the percentage of number 5 decreases the percentage of number 1,2 and 3 increases.In contrast to percentage of nymber 8 
#Which decreases as well.This is logical because they have similars shapes.
#We can use the above algorithm and to change each time the singular values and at the same time to observe the results
#It depends on the user wich value to use,if we want to be accurate in some specific number or to lose some others for example




#In this part we tune the algorithm as above but this time using only the firts singular values of each class

number_ones=[]
number_twos=[]
number_threes=[]
number_fours=[]
number_fives=[]
number_sixs=[]
number_sevens=[]
number_eights=[]
number_nines=[]
number_zeros=[]



for i in range(0,2006):
    norms=[]
    t=test_data.iloc[:,i]
    n1=n_1.iloc[:,0]
    n2=n_2.iloc[:,0]
    n3=n_3.iloc[:,0]
    n4=n_4.iloc[:,0]
    n5=n_5.iloc[:,0]
    n6=n_6.iloc[:,0]
    n7=n_7.iloc[:,0]
    n8=n_8.iloc[:,0]
    n9=n_9.iloc[:,0]
    n0=n_0.iloc[:,0]
    norm1=np.linalg.norm(t-n1)
    norms.append(norm1)
    norm2=np.linalg.norm(t-n2)
    norms.append(norm2)
    norm3=np.linalg.norm(t-n3)
    norms.append(norm3)
    norm4=np.linalg.norm(t-n4)
    norms.append(norm4)
    norm5=np.linalg.norm(t-n5)
    norms.append(norm5)
    norm6=np.linalg.norm(t-n6)
    norms.append(norm6)
    norm7=np.linalg.norm(t-n7)
    norms.append(norm7)
    norm8=np.linalg.norm(t-n8)
    norms.append(norm8)
    norm9=np.linalg.norm(t-n9)
    norms.append(norm9)
    norm0=np.linalg.norm(t-n0)
    norms.append(norm0)
    min_norm=min(norms)
    
    
    if min_norm==norm1:
        number_ones.append(test_data.iloc[:,i])
    elif min_norm==norm2:
        number_twos.append(test_data.iloc[:,i])
    elif min_norm==norm3:
        number_threes.append(test_data.iloc[:,i])
    elif min_norm==norm4:
        number_fours.append(test_data.iloc[:,i])  
    elif min_norm==norm5:
        number_fives.append(test_data.iloc[:,i])
    elif min_norm==norm6:
        number_sixs.append(test_data.iloc[:,i])
    elif min_norm==norm7:
        number_sevens.append(test_data.iloc[:,i])
    elif min_norm==norm8:
        number_eights.append(test_data.iloc[:,i])
    elif min_norm==norm9:
        number_nines.append(test_data.iloc[:,i])
    else:
        number_zeros.append(test_data.iloc[:,i])
        
classified_numbers=[len(number_zeros),len(number_ones),len(number_twos),len(number_threes),len(number_fours),len(number_fives),len(number_sixs),len(number_sevens),len(number_eights),len(number_nines)]
test_numbers=[len(test_0.columns),len(test_1.columns),len(test_2.columns),len(test_3.columns),len(test_4.columns),len(test_5.columns),len(test_6.columns),len(test_7.columns),len(test_8.columns),len(test_9.columns)]

percentage_of_numbers=[]

for i in range(10):
    percentage_of_numbers.append((classified_numbers[i]/test_numbers[i])*100)
    
    
percentage_of_success=pd.DataFrame(data=(classified_numbers,test_numbers,percentage_of_numbers))

percentage_of_success=percentage_of_success.transpose()
percentage_of_success.columns=['classified_numbers','test_numbers','percentage_of_numbers']
print(percentage_of_success)
        



#Now the results are : number 0 57.2%,number 1 150%,number 2 40%,number 3 103%
#number 4 53%,number 5 30%,number 6 157%,number 7 218%,number 8 56%,number 9 177%

#Tunning this algorithm we can see that only number  has a good prediction percentage. The variations from the other numbers
#is much more.Although this could be a good estimator for classifying the number in this task it isn't very efficient


#We coclude that GENERALLY THE NUMBERS ARE BAD WRITTEN.Only when the numbers are well written can be classified 
#more efficiently with the first sinular value.



#We can see this by plotting some of the numbers that have the biggest variation from 100%
#Plotiing only the vector of the first value

k=np.matrix(n_0.iloc[:,0])
k=k.reshape(16,16)
figure=plt.figure(figsize=(6,6))
plt.imshow(k)

k=np.matrix(n_1.iloc[:,0])
k=k.reshape(16,16)
figure=plt.figure(figsize=(6,6))
plt.imshow(k)

k=np.matrix(n_9.iloc[:,0])
k=k.reshape(16,16)
figure=plt.figure(figsize=(6,6))
plt.imshow(k)


k=np.matrix(n_6.iloc[:,0])
k=k.reshape(16,16)
figure=plt.figure(figsize=(6,6))
plt.imshow(k)

#We observe that indeed they are very bad Written






































