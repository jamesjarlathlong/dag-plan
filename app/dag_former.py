import functools
import app.helper as helper
from app.node_emulator import job_profiler

def var_namer(num, prefix):
    return prefix+'_'+str(num)
def add_sensers(sensers, mappers, reducers,graph):
    sensers = {k: {'children':[mappers[idx]]} for idx, k in enumerate(sensers)}
    return helper.merge_two_dicts(graph, sensers)
def add_mappers(sensers, mappers, reducers,graph):
    mappers = {k:{'children':reducers, 'parents':sensers[idx]} for idx, k in enumerate(mappers)}
    return helper.merge_two_dicts(graph, mappers)
def add_reducers(sensers, mappers, reducers,graph):
    reducers = {k: {'children':['K_0'],'parents':mappers} for idx, k in enumerate(reducers)}
    return helper.merge_two_dicts(graph, reducers)
def add_sink(sensers, mappers, reducers, graph):
    sink = {'K_0':{'children':[], 'parents':reducers}}
    return helper.merge_two_dicts(graph, sink)
def get_num_sensers(code_class):
    return len(get_allowed(code_class, 'sensenodes'))
def get_num_reducers(code_class):
    return len(get_allowed(code_class, 'reducenodes'))
def node_type(node_name):
    return node_name.split('_')[0]
def get_let(k):
    return k.split('_')[0]
def get_num(k):
    return int(k.split('_')[1])
def get_allowed(code_class, prop):
    return getattr(code_class, prop)
def generate_constraints(code, graph):
    code_class = define_the_class(code)
    sink = 0
    sensers = get_allowed(code_class, 'sensenodes')
    mappers = get_allowed(code_class, 'mapnodes')
    reducers = get_allowed(code_class, 'reducenodes')
    type_key = {'S':sensers, 'M':mappers, 'R':reducers, 'K':[[0]]}
    constraints = {k: type_key[get_let(k)][get_num(k)] for k in graph}
    return constraints
def generate_graph_structure(code_class):
    num_sensors = get_num_sensers(code_class)
    num_reducers = get_num_reducers(code_class)
    sensers = [var_namer(i, 'S') for i in range(num_sensors)]
    mappers = [var_namer(i, 'M') for i in range(num_sensors)]
    reducers = [var_namer(i, 'R') for i in range(num_reducers)]
    chain = [add_sensers, add_mappers, add_reducers, add_sink]
    paramed_chain = [functools.partial(i, sensers, mappers, reducers) for i in chain]
    init_graph = {}
    graph = helper.pipe(*paramed_chain)(init_graph)
    return graph

def define_the_class(code):
    exec(code)
    return locals()['SenseReduce']()
def get_weights(code_class):
    code_instance = code_class
    weights = job_profiler(code_instance)
    print('weights: ', weights)
    return weights
def add_weights_to_graph(weights_structure, graph):
    weights_structure['K'] = {0:{'cost':0, 'edge':{}}}
    for node, description in graph.items():
        node_level, node_idx = node.split('_')
        node_idx = int(node_idx)
        graph[node]['node_w'] = weights_structure[node_level][node_idx]['cost']
        graph[node]['edge_w'] = weights_structure[node_level][node_idx].get('edge',{})
    #new_graph = {k:helper.merge_two_dicts(v, weights[node_type(k)]) for k,v in graph.items()}
    return graph

def generate_weighted_graph(code):
    code_class = define_the_class(code)
    weight_adder = functools.partial(add_weights_to_graph, get_weights(code_class))
    full_graph_pipe = helper.pipe(generate_graph_structure, weight_adder)
    return full_graph_pipe(code_class)

