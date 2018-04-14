import random
import time
import app.helper as helper
import functools
import itertools
from copy import deepcopy
import json
from collections import defaultdict
from app import np
import random as urandom
import asyncio
import multiprocessing as mp
def hasher(i):
    prev={'58z': 23762, '22x': 28829, '56x': 23582, '22z': 28831, '95y': 36400, '53z': 23673, '53y': 23674, '53x': 23675, '95x': 36401, '95z': 36403, '41x': 22520, '41y': 22521, '41z': 22522, '68x': 24915, '68y': 24914, '68z': 24913, '39x': 30263, '39y': 30262, '39z': 30261, '69z': 24944, '15y': 27704, '15x': 27705, '55y': 23612, '55x': 23613, '15z': 27707, '55z': 23615, '63x': 24504, '63y': 24505, '63z': 24506, '43x': 22458, '43y': 22459, '43z': 22456, '32x': 29852, '21z': 29052, '32y': 29853, '21x': 29054, '21y': 29055, '17x': 27899, '61z': 24696, '17y': 27898, '61x': 24698, '61y': 24699, '17z': 27897, '46x': 22559, '46y': 22558, '46z': 22557, '32z': 29854, '37x': 29945, '37y': 29944, '37z': 29947, '64z': 24797, '96z': 36368, '64x': 24799, '96x': 36370, '96y': 36371, '64y': 24798, '40x': 22489, '40y': 22488, '40z': 22491, '69x': 24946, '69y': 24947, '49x': 22768, '49y': 22769, '49z': 22770, '18x': 28116, '18y': 28117, '18z': 28118, '56z': 23580, '22y': 28828, '58x': 23760, '56y': 23583, '58y': 23761}
    prevhash = prev.get(i, hash(i))
    return prevhash
def argmax(complex_list):
    abs_list = map(abs, complex_list)
    idx = abs_list.index(max(abs_list))
    return idx,complex_list[idx]
def timeit(method):
    def timed(*args, **kw):
        ts = time.clock()
        result = method(*args, **kw)
        te = time.clock()
        ex_time = te-ts
        return ex_time,result
    return timed
def avg_time(num_runs, method):
    def timed(*args, **kw):
        ts = time.clock()
        results = [method(*args, **kw) for i in range(num_runs)]
        te = time.clock()
        ex_time = (te-ts)/num_runs
        return ex_time, results[0]
    return timed
three_timer = functools.partial(avg_time, 3)
def time_s(method):
    def timed(*args, **kw):
        l = asyncio.get_event_loop()
        ts = time.time()
        res = l.run_until_complete(method(*args,**kw))
        te = time.time()
        ex_time = te-ts
        return ex_time,res
    return timed
def async_method(method):
    def asynced(*args, **kw):
        pool = mp.Pool(processes=1)
        res = pool.starmap(method, [args])[0]
        pool.close()
        pool.join()
        return res
    return asynced
def slowdown(n=1):
        def decorator(func):
            def wrapper(*args,**kw):
                n=1
                for i in range(n):
                    try:
                        res = [i for i in func(*args,**kw)]
                    except TypeError as e:
                        print('wasnt iterable, who cares')
                return func(*args, **kw)
            return wrapper
        return decorator
def bm1(size):
    def vec(size):
        return [urandom.getrandbits(8)/100 for i in range(size)]    
    mat = (vec(size) for i in range(size))
    v = np.Vector(*vec(size))
    t, res = three_timer(v.gen_matrix_mult)(mat)
    return t, res
def bm2(size):
    def vec(size):
        return [urandom.getrandbits(8)/100 for i in range(size)]    
    mat = [vec(size) for i in range(size)]
    t, res = three_timer(np.pagerank)(mat)
    return t, res
def bm3(size):
    def vec(size):
       return [urandom.getrandbits(8)/100 for i in range(size)]    
    v = vec(size)
    t, res = three_timer(np.fft)(v)
    return t, res
def bm4(size):
    def internal(size):
        if size<2:
            return size
        else:
            return bm4(size-1)+bm4(size-2)
    t, res = three_timer(internal)(size)
benchmark1 = bm1#async_method(bm1)
benchmark2 = bm2##async_method(bm2)
benchmark3 = bm3#async_method(bm3)
benchmark4 = bm4#async_method(bm4)
"""TODO - problem is that we are also counting the time to sample the fake data
for the benchmarks"""
class Node:
    def __init__(self,num):
        self.ID = num
        self.np = np
    def accelpacketyielder(self):
        return [random.uniform(-2,2) for i in range(120)]
    def accel(self, sample_length):
        """Read from the accelerometer a sample of length sample length.
           Accelerometer samples at 1000Hz, so if sample_length = 2000,
           the reading will be for 2 seconds."""
        result = {'x':[],'y':[],'z':[]}     
        while len(result['x'])<sample_length:
            packet = self.accelpacketyielder()
            ##print('len packet: ', len(packet))
            result['x'].extend(packet[0::3])
            result['y'].extend(packet[1::3])
            result['z'].extend(packet[2::3])
            ##print('result: ', len(result['x']))
        trimmed = {k:v[0:sample_length] for k,v in result.items()}
        yield from asyncio.sleep(0)
        return trimmed
    def testaccel(self, sample_length):
        fname = '/home/jjlong/dag_planner/dag-plan/app/192.168.123.31.json'
        print('opening: ',fname)
        with open(fname) as json_data:
            d = json.loads(json_data.read())
            yield from asyncio.sleep(0)
            trimmed = {k:v[0:sample_length] for k,v in d.items()}  
            return trimmed 
def execute_sampler(sampler, node):
    return sampler(node)
def run_sense(job_instance, node_idx, structure):
    node = Node(node_idx)
    time, data = time_s(execute_sampler)(job_instance.sampler, node)
    len_data = job_instance.l
    sampling_time = (len_data+random.uniform(0,10))/2000 #Hz
    structure['S'][node_idx]['cost'] = 1000*sampling_time #milliseconds
    structure['S'][node_idx]['edge'][node_idx] = data
    return structure
def execute_mapper(mapper, node, data):
    return [(k,v) for k,v in mapper(node, data)]
def partition(reducers, kv):
    idx = hasher(kv[0]) % len(reducers)
    return idx
def group_kvs_bykey(kvs):
    return {k:[i for i in pairs] for k,pairs in itertools.groupby(kvs, lambda x:x[0])}
def run_map(job_instance, node_idx, structure):
    node = Node(node_idx)
    data = structure['S'][node_idx]['edge'][node_idx]
    time, kvs = timeit(execute_mapper)(job_instance.mapper, node, data)
    btime,k = benchmark3(256)
    sorted_kvs = sorted(kvs, key = lambda x:x[0])
    hasher = functools.partial(partition, job_instance.reducenodes)
    grouped_by_hash = {k:list(g) for k,g in itertools.groupby(sorted_kvs, hasher)}
    structure['M'][node_idx]['cost'] = 1000*time #milliseconds
    for target, data in grouped_by_hash.items():
        structure['M'][node_idx]['edge'][target] = data
    return structure
def just_existing_values(d):
    return {k:v for k,v in d.items() if v}
def not_none(lst):
    return [i for i in lst if i is not None]

def get_in_data(structure, reducenode):
    map_struct = deepcopy(structure['M'])
    all_kvs = [map_struct[i]['edge'] for i in range(len(map_struct))]
    all_existing = [just_existing_values(d) for d in all_kvs]
    each_mappers_contribution = not_none([i.get(reducenode) for i in all_existing]) #to this reduce node
    flattened = itertools.chain(*each_mappers_contribution)
    srted = sorted(flattened, key =lambda x:x[0])
    grouped = {k:list(g) for k,g in itertools.groupby(srted, key = lambda x:x[0])}
    just_vals = {k:[kv[1] for kv in v] for k, v in grouped.items()}
    return just_vals
def execute_reducer(reducer, in_data):
    return [list(reducer(k,vs)) for k,vs in in_data.items()]
def run_reduce(job_instance, node_idx, structure):
    in_data = get_in_data(structure, node_idx)
    print(node_idx, ',',in_data)
    reduce_func = functools.partial(job_instance.reducer, node_idx)
    time, out_data = timeit(execute_reducer)(reduce_func, in_data)
    structure['R'][node_idx]['cost']= 1000*time #milliseconds
    structure['R'][node_idx]['edge'][0] = out_data
    return structure
def add_sense(job_instance, structure):
    num_s = len(job_instance.sensenodes)
    sense_structure = {'S': {i: {'cost':None, 'edge':{i:None}}  for i in range(num_s)}}
    return helper.merge_two_dicts(sense_structure, structure)
def add_map(job_instance, structure):
    num_m = len(job_instance.mapnodes)
    num_r = len(job_instance.reducenodes)
    map_structure = {'M': {i: {'cost':None, 'edge':{j:None for j in range(num_r)}} for i in range(num_m)}}
    return helper.merge_two_dicts(map_structure, structure)
def add_reduce(job_instance, structure):
    num_r = len(job_instance.reducenodes)
    red_structure = {'R': {i: {'cost':None, 'edge':{0:None}} for i in range(num_r)}}
    return helper.merge_two_dicts(red_structure, structure)
def get_profile_structure(job_instance):
    structure = {}
    adders = [functools.partial(add_sense, job_instance), functools.partial(add_map, job_instance), functools.partial(add_reduce, job_instance)]
    return helper.pipe(*adders)(structure)
def run_all(run_func, job_instance, nodes, structure):
    add_sensers = helper.pipe(*[functools.partial(run_func, job_instance, i) for i in range(len(nodes))])
    return add_sensers(structure)
def float_json(data):
    return json.dumps(data,separators=(',',':'))
def convert_to_lengths(final_struct):
    print('final_struct: ', final_struct['M'])
    for k,v in final_struct.items():
        for node, values in v.items():
            for target_node, msg in values['edge'].items():
                msg_length = len(float_json(msg)) if msg else 0
                values['edge'][target_node] = msg_length
    return final_struct
def job_profiler(job_instance):
    structure = get_profile_structure(job_instance)
    sense_struct = run_all(run_sense, job_instance, job_instance.sensenodes, structure)
    map_struct = run_all(run_map, job_instance, job_instance.mapnodes, sense_struct)
    reduce_struct = run_all(run_reduce, job_instance, job_instance.reducenodes, map_struct)
    just_lengths = convert_to_lengths(reduce_struct)
    return just_lengths



        

