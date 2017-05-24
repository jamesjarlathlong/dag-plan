import requests
code = """class SenseReduce:
    def __init__(self):
        self.sensenodes = [[1],[2]]
        self.mapnodes = [[0,1,2,3],[0,1,2,3]]
        self.reducenodes = [[0,1,2]]
    def sampler(self,node):
        acc = yield from node.accel(20000) ###a lot of data
        return acc
    def mapper(self,node,d):
        import time
        avgs = {k: round(sum(v)/len(v),2) 
                for k,v in d.items()} #just avg
        yield(node.ID,avgs)#a much smaller amount of data
    def reducer(self,node,k,vs):
        yield(k,vs) #no reduction from map"""
r = requests.post("http://0.0.0.0:5000/solve", data={'code':code})
print('r: ', r.text)