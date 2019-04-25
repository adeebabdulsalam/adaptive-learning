import pickle


def switch(grade):
    if grade == "S":
        return 10
    elif grade == "A":
        return 9
    elif grade == "B":
        return 8
    elif grade == "C":
        return 7
    elif grade == "D":
        return 6
    elif grade == "E":
        return 5
    else:
        return 0


X = [['B', 'B', 'C', 'B']]  # Input is here
for i in X:
    for j in range(0,4):
        i[j]=switch(i[j])

kmeans = pickle.load(open('saved/kmeans.sav', 'rb'))


Y = kmeans.predict(X)
print(Y)
