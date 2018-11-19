import json


def read_txt_lines(path):
    lines = []
    with open(path,'r',encoding="utf-8") as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    return lines

def write_csv(path,headers,items):
    with open(path,'w',encoding="utf-8") as f:
        print(",".join(map(str,headers)),file=f)
        for features in items:
            print(",".join(map(str,features)),file=f)


#def write_txt_lines(path,items):
#    lines = []
#    with open(path,'w',encoding="utf-8") as f:
#        for row in items:
#            print(srow+"",file=f)
#    return lines

def read_json_utf8(path):
    with open(path,'r',encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def write_json_utf8(path,obj):
    with open(path,'w',encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)


