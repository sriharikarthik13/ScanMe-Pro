import random
listofemployeeid=[]

valuesofemployeeid = ['10000','2345','55555']

for i in valuesofemployeeid:
    listofemployeeid.append(i)


employeeidval = "0"
while employeeidval not in listofemployeeid:
    random_num = random.randint(1000,1002)
    if random_num not in listofemployeeid:
        employeeidval = str(random_num)
        break

print(employeeidval)