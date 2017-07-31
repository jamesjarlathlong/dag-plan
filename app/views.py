from app import app
#from models import Result
from flask import (Flask, request, Response)
import json
import functools
import app.dag_former as dag_former
import app.dag_solver as dag_solver
import app.bandwidth_calculator as bandwidth
import app.node_emulator as node_emulator
import math
import itertools
import collections
def inted(dct):
    return {int(k):v for k,v in dct.items()}
def load_rssi(jsonified):
    j = json.loads(jsonified)
    return inted({k:inted(v) for k,v in j.items()})
def load_px(jsonified):
    j = json.loads(jsonified)
    return inted(j)
@app.route('/solve', methods=['POST'])
def solve():
    code = request.form['code']
    rssi = load_rssi(request.form['rssi'])
    px = load_px(request.form['px'])
    print('p,r',px,rssi)
    solution = solve_LP(code,rssi=rssi,proc=px)
    print('sol: ', solution)
    return Response(json.dumps(solution), status = 200)
def solve_LP(code, rssi=None, proc=None):
    graph = dag_former.generate_weighted_graph(code)
    total_num = 10
    if not rssi:
        rssi = create_rssi(total_num)
    if not proc:
        proc = create_processors(rssi)
    solution= dag_solver.solve_DAG(code,rssi,proc)
    return solution

def scale_proc(d):
    """takes a list of dictionaries of the form [{node:91, size:10, t:0.03}]"""
    sizes = set([i['size'] for i in d])
    ref = {k:node_emulator.benchmark1(k)[0] for k in sizes}
    def scale_d_item(d_item, ref):
        ref_time = ref[d_item['size']]
        return ref_time/d_item['t']
    proc = {k['node']: scale_d_item(k, ref) for k in d}
    return proc


def create_processors(rssi):
    ks = [95,0,31]
    #return {k:1 if k==0 else 0.05 for k in range(len(rssi))}
    return {k:1 if k==0 else 0.05 for k in ks}
def create_rssi(total_num):
    #rssi = create_network(range(3), range(3,12),1)
    #return mirror(rssi)
    rssi = {95:{31:-50,0:-50,},31:{95:-50,0:-20},0:{95:-45}}
    return rssi
def chunk(lst, num_chunks):
    chunk_size = math.ceil(len(lst)/num_chunks)
    chunks = (list(itertools.islice(lst, x, x+chunk_size))
              for x in range(0, len(lst), chunk_size))
    return list(chunks)
def lst_to_dict(lst):
    return {k:-20 for k in lst}
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
def assign_edge(edges, routers, degree):
    num_buckets = len(routers)
    chunks = chunk(edges, num_buckets)
    adjacency = {k:[] for k in edges}
    bucketiser = functools.partial(buckets, num_buckets,degree)
    idx_to_node = functools.partial(translater,routers)
    for chk in chunks:
        for node in chk:
            adjacency[node]+=idx_to_node(bucketiser(node))
    return {k:lst_to_dict(v) for k,v in adjacency.items()}
def mirror(d):
    new_d = collections.defaultdict(dict)
    for k,v in d.items():
        for node, weight in v.items():
            new_d[k][node] = weight
            new_d[node][k] = weight
    return new_d
def translater(lst, idxs):
    return [lst[i] for i in idxs]
def buckets(num_buckets, degree, idx):
    return [(idx+offset)%num_buckets for offset in range(degree)]
def create_network(routers, edges, edge_degree):
    def adjacent(node, possibles):
        return {k:-50 for k in possibles if k!=node}
    router_net = assign_edge(routers, routers, 2)
    edge_to_router = assign_edge(edges, routers, edge_degree)
    return merge_two_dicts(router_net, edge_to_router)

