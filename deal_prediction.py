import numpy

def read_data_from_set(file_name):
    list = []
    fout = open("testtest.txt", "w")
    with open(file_name, 'r') as dataset:
        for index, elem in enumerate(dataset):
            tmp = elem.replace("[","").replace("]","").replace("\n","").split(" ")
            list.extend(tmp)
            if elem[-2] == "]":
                list1 = [x.strip(" ") for x in list if x]
                list2 = ",".join(list1)
                fout.write(list2+"\n")
                list = []
    fout.close()
  
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False             
            


if __name__ == "__main__":
    read_data_from_set("prediction.txt")
