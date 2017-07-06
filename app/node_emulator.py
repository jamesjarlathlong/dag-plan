import random
import time
import app.helper as helper
import functools
import itertools
from copy import deepcopy
import json
from collections import defaultdict
import app.np as np
import random as urandom
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        ex_time = te-ts
        return ex_time,result
    return timed
@timeit
def benchmark1(size):
    def vec(size):
        return [urandom.getrandbits(8)/100 for i in range(size)]    
    mat = (vec(size) for i in range(size))
    v = np.Vector(*vec(size))
    res = v.gen_matrix_mult(mat)
    return 
class Node:
    def __init__(self,num):
        self.ID = num
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
        yield trimmed
def execute_sampler(sampler, node):
    return [d for d in sampler(node)][0]
def run_sense(job_instance, node_idx, structure):
    node = Node(node_idx)
    time, data = timeit(execute_sampler)(job_instance.sampler, node)
    structure['S'][node_idx]['cost'] = len(data['x'])+random.uniform(0,250)
    structure['S'][node_idx]['edge'][node_idx] = data
    return structure
def execute_mapper(mapper, node, data):
    return [(k,v) for k,v in mapper(node, data)]
def partition(reducers, kv):
    idx = hash(kv[0]) % len(reducers)
    return idx
def group_kvs_bykey(kvs):
    return {k:[i[1] for i in pairs] for k,pairs in itertools.groupby(kvs, lambda x:x[0])}
def run_map(job_instance, node_idx, structure):
    node = Node(node_idx)
    data = structure['S'][node_idx]['edge'][node_idx]
    time, kvs = timeit(execute_mapper)(job_instance.mapper, node, data)
    sorted_kvs = sorted(kvs, key = lambda x:x[0])
    hasher = functools.partial(partition, job_instance.reducenodes)
    grouped_by_hash = {k:group_kvs_bykey(g) for k,g in itertools.groupby(sorted_kvs, hasher)}
    structure['M'][node_idx]['cost'] = time
    for target, data in grouped_by_hash.items():
        structure['M'][node_idx]['edge'][target] = data
    return structure
def just_existing_values(d):
    return {k:v for k,v in d.items() if v}
def not_none(lst):
    return [i for i in lst if i is not None]
def merge_dicts_of_lists(d1, d2):
    d = d1.copy()
    dd = defaultdict(list,d)
    for i,j in d2.items():
        dd[i].extend(j)
    return dict(dd)
def get_in_data(structure, reducenode):
    map_struct = deepcopy(structure['M'])
    all_kvs = [map_struct[i]['edge'] for i in range(len(map_struct))]
    all_existing = [just_existing_values(d) for d in all_kvs]
    for_reduce_node = not_none([i.get(reducenode) for i in all_existing])
    return functools.reduce(merge_dicts_of_lists, for_reduce_node) 
def execute_reducer(reducer, in_data):
    return [list(reducer(k,vs)) for k,vs in in_data.items()]
def run_reduce(job_instance, node_idx, structure):
    in_data = get_in_data(structure, node_idx)
    reduce_func = functools.partial(job_instance.reducer, node_idx)
    time, out_data = timeit(execute_reducer)(reduce_func, in_data)
    structure['R'][node_idx]['cost']=time
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
def convert_to_lengths(final_struct):
    for k,v in final_struct.items():
        for node, values in v.items():
            for target_node, msg in values['edge'].items():
                msg_length = len(json.dumps(msg)) if msg else 0
                values['edge'][target_node] = msg_length
    return final_struct
def job_profiler(job_instance):
    structure = get_profile_structure(job_instance)
    sense_struct = run_all(run_sense, job_instance, job_instance.sensenodes, structure)
    map_struct = run_all(run_map, job_instance, job_instance.mapnodes, sense_struct)
    reduce_struct = run_all(run_reduce, job_instance, job_instance.reducenodes, map_struct)
    just_lengths = convert_to_lengths(reduce_struct)
    return just_lengths



        

