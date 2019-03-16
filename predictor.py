import numpy as np
import pickle


svclassifier = pickle.load(open("prediction-svm3.sav",'rb'))

#input array
X=[1,0.99,0.99,1400,1,1,1,1,1,0]
#n indicates number of features in the input
n=10
      # "Enter input array as"
      # "[Gender:Male=>1 ; Female=>0]"
      # "[CGPA]"
      # "[12th grade]"
      # "[Codechef Rating]"
      # "[Internship: Yes=>1 ; No=>0]"
      # "[OS,Networks,DBMS,DSA : S=>10 - - - E=>5]"
      # "[Interviews attempted]"
#
# for i in range(n):
#     print("Enter feature {}:".format(i+1))
#     x=input()
#     if i == 1 or i == 5:
#         x = int(x)/10
#     if i == 2:
#         x = int(x)/100
#     X.append(int(x))

X=np.reshape(X,(-1,int(n)))
y=svclassifier.predict(X)

print("Predicted Tier: ", y)
