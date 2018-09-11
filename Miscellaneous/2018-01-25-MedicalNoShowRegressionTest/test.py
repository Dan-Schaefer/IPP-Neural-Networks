import pandas as pds

#import os
#print(os.getcwd())
#os.chdir('C:\\Users\\username\\Desktop\\headfirstpython\\chapter3')

# your current working directory will be displayed
#os.path.exists('testfile.csv')
# Must be True, otherwise it is unrelated to pandas


dataframeX = pds.read_csv("C:\\Users\\danie\\Documents\\Masterarbeit-IPP\\Python\\Keras\\2018-01-25-MedicalNoShowRegressionTest\\No-show-Issue-Comma-300k.csv", usecols=[0, 1, 4, 6, 7, 8, 9, 10, 11, 12, 13])
print(dataframeX)
