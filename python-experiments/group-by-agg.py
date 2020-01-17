## Python code written for self learning by Chaitanya Kulkarni

#Read and process data from CSV file
from os import path
import csv
def validate_number(x):
    try:
        return float(x)
    except:
        return 0
csvfile = path.join("C:\BigDataApps\sample-data\input\india-trade-data","2018-2010_export.csv")
with open(csvfile) as csvdata:
    c = 0
    rows = csv.DictReader(csvdata, delimiter = ',')
    print(rows.fieldnames)

    for row in rows:
        c = c +1
# print total number of rows        
    print(c)
    
#aggregate for a column from the csv data
with open(csvfile) as csvdata:
    c = 0
    rows = csv.DictReader(csvdata, delimiter = ',')
    print(rows.fieldnames)

    for row in rows:
        c = c +  validate_number(row["value"])
    print(c)

grpdict = {}

#apply a group by like functionality
with open(csvfile) as csvdata:
    c = 0
    rows = csv.DictReader(csvdata, delimiter = ',')
    print(rows.fieldnames)
    grpdict = {}
    for row in rows:
        cmdty = row["Commodity"]
        #if row["Commodity"] in grpdict:
        newval = validate_number(grpdict.get(cmdty)) + validate_number(row["value"])
        grpdict[cmdty] = round(newval,2)
        #grpdict.update(cmdty=newval)

        #c = c +  validate_number(row["value"])
    print(grpdict)
    



