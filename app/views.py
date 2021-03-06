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
def not_there(d, nodelist):
    return [i for i in nodelist if i not in d]
def get_opp(source, target, graph):
    return (target,graph[target].get(source))
def get_opposites(missing, graph):
    possiblevals = {k:[get_opp(k,v,graph) for v in vs]
            for k,vs in missing.items()}
    nonones = {k:dict([i for i in vs if i[1]])
               for k,vs in possiblevals.items()}
    return {k:v for k,v in nonones.items() if v}
def two_way(graph):
    all_nodes = [k for k in graph]
    missing_oneway = {k:not_there(v, all_nodes)
                     for k,v in graph.items()}
    opps = get_opposites(missing_oneway,graph)
    for source,targets in opps.items():
        for target, value in targets.items():
            graph[source][target] =value
    return graph
def load_rssi(jsonified):
    j = json.loads(jsonified)
    #twoed = two_way(j)
    return inted({k:inted(v) for k,v in j.items()})
def load_px(jsonified):
    j = json.loads(jsonified)
    print('px: ',j)
    existing = {k:v for k,v in j.items() if v.get('t')}
    # run it and scale everything by it. - need to ensure
    #that jsonified is of the form: {91:{size:10, t:0.03}, ..
    scaled_j = scale_proc(existing)
    print('scaled: ', scaled_j)
    inte = inted(scaled_j)
    print('inted: ', inte)
    return inte
def load_ack(jsonified):
    a = json.loads(jsonified)
    ack = float(a.get('t'))
    return ack
@app.route('/benchmark', methods = ['POST'])
def bench():
    size = int(request.form['size'])
    cat = request.form['cat']
    cat_dict = {'b1':node_emulator.benchmark1
               ,'b2':node_emulator.benchmark2
               ,'b3':node_emulator.benchmark3
               ,'b4':node_emulator.benchmark4}
    res = cat_dict[cat](size)
    print('res: ', res)
    return Response(json.dumps({'t':res[0]}), status = 200)
@app.route('/solve', methods=['POST'])
def solve():
    code = request.form['code']
    rssi = load_rssi(request.form['rssi'])
    px = load_px(request.form['px'])
    ack = load_ack(request.form['ack'])
    print('p,r',px,rssi)
    solution = solve_LP(code,rssi=rssi,proc=px,ack=ack)
    print('sol: ', solution)
    return Response(json.dumps(solution), status = 200)
def solve_LP(code, rssi=None, proc=None, ack=580):
    graph = dag_former.generate_weighted_graph(code)
    total_num = 10
    if not rssi:
        rssi = create_rssi(total_num)
    if not proc:
        proc = create_processors(rssi)
    solution= dag_solver.solve_DAG(code,rssi,proc,ack)
    return solution

def scale_proc(d):
    """takes a list of dictionaries of the form {91:{size:10, t:0.03},.."""
    sizes = set([(v['size'], v['cat']) for k,v in d.items()])
    f_map = {'b1':node_emulator.benchmark1, 'b2':node_emulator.benchmark2
            ,'b3':node_emulator.benchmark3,'b4':node_emulator.benchmark4}
    ref = {k:f_map[k[1]](k[0])[0] for k in sizes}
    print('reference times: ', ref)
    def scale_d_item(d_item, ref):
        ref_time = ref[(d_item['size'], d_item['cat'])]
        return ref_time/d_item['t']
    proc = {k: scale_d_item(v, ref) for k,v in d.items()}
    proc['0'] = 1
    print('proc: ', proc)
    return proc


def create_processors(rssi):
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

