import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()


data_file = open('data_for_test.txt','r')
#check_file = open('check.txt','w')
lines = data_file.readlines()
x=[]
y=[]

for line in lines:
    data_line = line.split(' ')
    print (data_line)
    x.append(data_line[4])
    y.append(data_line[5])
data_file.close()
# check_file.close()
plt.plot(x,y)
plt.show()


