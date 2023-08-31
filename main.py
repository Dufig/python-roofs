import openpyscad as ops

f = open("kvadr.scad", "r")
f = f.read()

def find_list(f: str) -> list:
    f = f[:f.index(";")]
    print(f)


Points = find_list(f)





if __name__ == '__main__':
    pass
