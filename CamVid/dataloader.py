# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:36:04 2022

@author: Shefali Garg
"""

# importing csv module
import csv

# csv file name
filename = "class_dict.csv"

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open(filename, 'r') as csvfile:
	# creating a csv reader object
	csvreader = csv.reader(csvfile)
    
    
            
    # extracting field names through first row
	fields = next(csvreader)

	# extracting each data row one by one
	for row in csvreader:
		rows.append(row)
        
	# get total number of rows
	print("Total no. of rows: %d"%(csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))

#label = rows[:1]
#print(label)
#print(rows)


for row in rows[:32]:
	# parsing each column of a row
	for col in row:
		print("%10s"%col),
	print('\n')
