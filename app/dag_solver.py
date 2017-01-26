import itertools
import collections
import random
import pulp
import json
import functools
import time
import random
import app.helper as helper
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r %f sec' %(method.__name__, te-ts))
        return result
    return timed
def find_communication_power(a, b, rssi):
    """a and b are the names of processor"""
    return rssi[a][b]
def find_computation_power(a, processors):
    return processors[a]
def dummy_namer(tupl):
    template = collections.namedtuple('Assignment', 'edge parent child')
    return template(edge=tupl[0], parent=tupl[1], child=tupl[2])
def lazy_flattener(listoflists):
    return itertools.chain(*listoflists)
def find_node_edges(nodekey,nodevalue):
    return itertools.product([nodekey], nodevalue['children'])
def find_graph_edges(graph):
    """given a dictionary representation of a graph
    generate a list of the graph edges as tuples"""
    nested_combinations = (find_node_edges(k,v) for k,v in graph.items())
    return lazy_flattener(nested_combinations)
def find_edgeparents(edge,constraints):
    """given an edge from graph, find valid parent processors given constraints"""
    parent_node = edge[0]
    return constraints[parent_node]
def find_edgechildren(edge,constraints):
    """given an edge from graph, find valid parent processors given constraints"""
    parent_node = edge[1]
    return constraints[parent_node]
def combination_finder(edge, constraints):

    return ([edge], find_edgeparents(edge,constraints), find_edgechildren(edge,constraints))
def timed_product(*args):
    return itertools.product(*args)
@timeit
def unroll(combination_generator):
    return (dummy_namer(comb) for product in combination_generator
            for comb in product)
@timeit
def generate_dummies(graph, constraints):
    """generate a dummy variable named tuple for each
    valid combination of edge, assigned parent, assigned child"""
    edges = find_graph_edges(graph)
    edge_possibilities = (combination_finder(edge,constraints) for edge in edges)
    combination_generator = (timed_product(*edge) for edge in edge_possibilities)
    all_dummies = unroll(combination_generator)
    return all_dummies
def add_cost_function(problem, dummy,dummy_vars, cost_calculator):
    cost_function = (cost_calculator(i)*dummy_vars[i] for i in dummy)
    problem += pulp.lpSum(cost_function), "Sum of DAG edge weights"
    return problem
def find_communication_power(a, b, rssi):
    """a and b are the names of processor"""
    return rssi[a][b]
def find_computation_power(a, processors):
    return processors[a]
def find_cost(graph, processors, rssi, a_dummy):
    """node is a key value pair, assignments is a named tuple"""
    parent_node = graph[a_dummy.edge[0]]
    child_node = graph[a_dummy.edge[1]]
    parent = a_dummy.parent
    child = a_dummy.child
    comp_cost = parent_node['node_w']/processors[parent]
    comm_cost_next = child_node['edge_w']/rssi[parent][child]
    return comp_cost+comm_cost_next
@timeit
def edge_uniqueness(problem, dummies, dummy_vars):
    """given all possible dummy variable assignments
    and the ILP problem, create the constraints that 
    guarantee only one dummy variable is turned on
    for each edge"""
    def get_edge(tupl):
        return tupl.edge
    grouped = (g for k,g in itertools.groupby(dummies, get_edge))
    for group in grouped:
        problem = constrain_one_edge(problem, group, dummy_vars)
    return problem
def constrain_one_edge(problem, grouped_by_edge, dummy_vars):
    """given a list of dummy variables corresponding to an edge
    generate constraint statement for each edge e.g x+y+z <=1, -x-y-z <= 1
    """
    edge_vars = (dummy_vars[i] for i in grouped_by_edge)
    problem += (pulp.lpSum(edge_vars)==1,
                "sum of dummies eq 1 for: "+str(random.random()))
    return problem
###=======make sure edge assignments match at nodes======###
def match_parentchild(edge, edges):
    """find any neighbouring edges of node"""
    return ((edge,i) for i in edges if i[0] == edge[1])
def find_neighboring_edges(graph):
    """find all pairs of edges where child edge_i = parent edge_j"""
    edges = find_graph_edges(graph)
    return lazy_flattener((match_parentchild(edge, edges) for edge in edges))
def inconsistent_with_one(in_edge_assignment, all_out_edges):
    """given an assignment corresponding to the edge into a given
    node, find all assignments corresponding to the outward edge
    where inward_child!=outward_parent"""
    return ((in_edge_assignment,i) for i in all_out_edges
            if i.parent!=in_edge_assignment.child)
def edgepair_inconsistents(dummies, in_edge, out_edge):
    """given an in edge, e.g. ('A','D'), and an out edge, e.g.
    ('D', 'G') find all dummy assignments pairs where
    in_edge assignment child does not equals out_edge assignment parent
    this is basically just the inconsistent_with_one function, but applied
    to each possible assignment for the inward edge and then combined back
    into a single list
    """
    matching_in_edge = (i for i in dummies if i.edge == in_edge)
    matching_out_edge = [i for i in dummies if i.edge == out_edge]
    return lazy_flattener((inconsistent_with_one(i, matching_out_edge)
                      for i in matching_in_edge))
def all_inconsistents(graph, dummies):
    """this function applies the find_inconsistent_assignments function
    over the whole graph: first by finding all pairs of in_edge, out_edge
    and then simply applying the function to each of these pairs in turn"""
    edge_pairs = find_neighboring_edges(graph)
    catcher = functools.partial(edgepair_inconsistents, dummies)
    wrong_nodes = lazy_flattener((catcher(*i) for i in edge_pairs))
    return wrong_nodes
@timeit
def inout_consistency(graph, dummies, problem, dummy_vars):
    all_matchers = all_inconsistents(graph, dummies)
    for inconsistent_pair in all_matchers:
        description = json.dumps(inconsistent_pair)
        added_dummy_vars = [dummy_vars[i] for i in inconsistent_pair]
        problem += (pulp.lpSum(added_dummy_vars)<=1,
        "pick one of mutex pair: "+description)
    return problem
@timeit
def create_list_from_gen(gen):
    return list(gen)
@timeit
def formulate_LP(graph, constraints, processors, rssi):
    d_gen = functools.partial(generate_dummies, graph, constraints)
    d = create_list_from_gen(d_gen())
    problem = pulp.LpProblem("DAG",pulp.LpMinimize)
    dummy_vars = pulp.LpVariable.dicts("Sensor", d,0, 1, 'Binary')
    cost_calculator = functools.partial(find_cost, graph, processors, rssi)
    problem = add_cost_function(problem, d, dummy_vars, cost_calculator)
    problem = edge_uniqueness(problem, d, dummy_vars)
    problem = inout_consistency(graph, d_gen(), problem, dummy_vars)
    return problem
@timeit
def solver(p):
    res = p.solve()
    return p
def to_tuples(str):
    no_unders = str.replace(',_',', ')
    Sensor_Assignment = collections.namedtuple('Sensor_Assignment', 'edge parent child')
    exec('tpl = '+no_unders)
    return locals()['tpl']
def get_val(tupl):
    return tupl[1]
def keyfunct(tupl):
    return tupl[0].split(',_parent')[0]
def get_level(node_assignment_tpl):
    level = node_assignment_tpl[0].split('_')[0]
    print('level: ', level)
    return level
def get_node_order(tpl):
    return tpl[0].split('_')[1]
def sortbynode(lst):
    return sorted(lst, key=get_node_order)
def tuple_to_kv(tpl):
    edge = tpl.edge
    yield (edge[0], tpl.parent)
    yield (edge[1], tpl.child)
def tuples_to_node_assignment_pairs(tuples):
    all_tuples = lazy_flattener(tuple_to_kv(tpl) for tpl in tuples)
    node_assignment_pairs = {k:v for k,v in all_tuples}
    print('pairs: ', node_assignment_pairs)
    return sorted([(k,v) for k,v in node_assignment_pairs.items()], key=get_level)
def output(solution):
    all_nonzero = [(v.name,v.varValue) for v in solution.variables() if v.varValue >0]
    grouped = [list(g) for k,g in itertools.groupby(all_nonzero, keyfunct)]
    chosen = [max(i, key = get_val) for i in grouped]
    converted = [to_tuples(i[0]) for i in chosen]
    node_assignment_pairs = tuples_to_node_assignment_pairs(converted)
    print('node_assignment_pairs: ', node_assignment_pairs)
    grouped_by_level = {k:list(g) for k,g in itertools.groupby(node_assignment_pairs, get_level)}
    print('grouped: ', grouped_by_level)
    chosen = {k: [i[1] for i in sortbynode(g)] for k,g in grouped_by_level.items()}
    return chosen

solution_pipe = helper.pipe(formulate_LP, solver, output)

