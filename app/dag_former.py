import functools
def compose_two(func1, func2):
     def composition(*args, **kwargs):
        return func1(func2(*args, **kwargs))
     return composition
def compose(*funcs):
    return functools.reduce(compose_two, funcs)
def pipe(*funcs):
    return functools.reduce(compose_two, reversed(funcs))

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z
def var_namer(num, prefix):
    return prefix+'_'+str(num)
def add_sensers(sensers, mappers, reducers,graph):
    sensers = {k: {'children':[mappers[idx]]} for idx, k in enumerate(sensers)}
    return merge_two_dicts(graph, sensers)
def add_mappers(sensers, mappers, reducers,graph):
    mappers = {k:{'children':reducers, 'parents':sensers[idx]} for idx, k in enumerate(mappers)}
    return merge_two_dicts(graph, mappers)
def add_reducers(sensers, mappers, reducers,graph):
    reducers = {k: {'children':['K_0'],'parents':mappers} for idx, k in enumerate(reducers)}
    return merge_two_dicts(graph, reducers)
def add_sink(sensers, mappers, reducers, graph):
    sink = {'K_0':{'children':[], 'parents':reducers}}
    return merge_two_dicts(graph, sink)
def get_num_sensers(code_class):
    return len(get_allowed(code_class, 'sensenodes'))
def get_num_reducers(code_class):
    return len(get_allowed(code_class, 'reducenodes'))
def profile_cost(function):
    return {'node_w':10, 'edge_w':4}
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
    graph = pipe(*paramed_chain)(init_graph)
    return graph

class dummy:
    def __init__(self):
        self.sensenodes = [[0,1], [1,4]]
        self.mapnodes = [[0,1,3], [1,4,5]]
        self.reducenodes = [[0], [1]]
    def senser(self):
        pass
    def mapper(self):
        pass
    def reducer(self):
        pass
def define_the_class(code):
    dummy_class = dummy()
    return dummy_class
def get_weights(code_class):
    steps = {'S':code_class.senser, 'M':code_class.mapper, 'R':code_class.reducer, 'K':lambda: None}
    weights = {k:profile_cost(v) for k,v in steps.items()}
    return weights
def add_weights_to_graph(weights, graph):
    new_graph = {k:merge_two_dicts(v, weights[node_type(k)]) for k,v in graph.items()}
    return new_graph

def generate_weighted_graph(code):
    code_class = define_the_class(code)
    weight_adder = functools.partial(add_weights_to_graph, get_weights(code_class))
    full_graph_pipe = pipe(generate_graph_structure, weight_adder)
    return full_graph_pipe(code_class)
