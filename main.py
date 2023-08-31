import openpyscad as ops

f = open("kvadr.scad", "r")
f = f.read()

def find_list(no_list: str) -> list:
    no_list = no_list[:no_list.index(";")]
    no_list = no_list[no_list.index("["):]
    return list(no_list)


Points = list(find_list(f))
print(type(Points))





if __name__ == '__main__':
    pass
