import itertools
import collections
import random
import pulp
import json
import functools
import time
import random
import app.helper as helper
import app.np as np
import app.dag_former as dag_former
import app.bandwidth_calculator as bandwidth
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
def unroll(combination_generator):
    return (dummy_namer(comb) for product in combination_generator
            for comb in product)
def generate_dummies(graph, constraints):
    """generate a dummy variable named tuple for each
    valid combination of edge, assigned parent, assigned child"""
    edges = list(find_graph_edges(graph))
    edge_possibilities = [combination_finder(edge,constraints) for edge in edges]
    combination_generator = (timed_product(*edge) for edge in edge_possibilities)
    all_dummies = unroll(combination_generator)
    return all_dummies
def get_stage(dummy):
    """takes a dummy var namedtuple, e.g. Assignment(edge = ('S_0', 'M_0'), parent =0, child=1)
    and returs the stage identifier of the child, in this instance M"""
    return dummy.edge[1].split('_')[0]
def group_by_stage(dummies):
    sorted_dummies = sorted(dummies, key = get_stage)
    return (g for k,g in itertools.groupby(sorted_dummies, get_stage))
def is_bottleneck_pair(dummy_one, dummy_two):
    same_child = dummy_one.child==dummy_two.child
    different_edge = dummy_one.edge!=dummy_two.edge
    return same_child and different_edge
def find_stage_bottleneck_pairs(stage_dummies):
    all_pairs = itertools.combinations(stage_dummies, 2)
    return (pair for pair in all_pairs if is_bottleneck_pair(*pair))
def find_all_bottleneck_pairs(dummies):
    all_bottlenecks = (find_stage_bottleneck_pairs(stage_dummies) 
                        for stage_dummies in group_by_stage(dummies))
    return helper.lazy_flattener(all_bottlenecks)
def logical_and(dependent, driver1, driver2):
    return [dependent>=driver1+driver2-1,
            dependent<=driver1,
            dependent<=driver2]
def add_constraints(problem, constraints):
    for c in constraints:
        problem += c
    return problem
def add_bottleneck_constraint(bottleneck_vars, dummy_vars, problem, bottleneck_dummy):
    dependent = bottleneck_vars[bottleneck_dummy]
    driver1 = dummy_vars[bottleneck_dummy.one]
    driver2 = dummy_vars[bottleneck_dummy.two]
    constraints = logical_and(dependent, driver1, driver2)
    problem = add_constraints(problem, constraints)
    return problem
def constrain_bottlenecks(bottleneck_vars, dummy_vars, problem, bottleneck_dummies):
    constrainer = functools.partial(add_bottleneck_constraint, bottleneck_vars, dummy_vars)
    for dummy in bottleneck_dummies:
        problem = constrainer(problem, dummy)
    return problem
def create_bottleneck_dummy(dummy_one, dummy_two):
    template = collections.namedtuple('Bottleneck', 'one two')
    return template(one=dummy_one, two=dummy_two)
def create_bottleneck_dummies(pairs):
    return (create_bottleneck_dummy(*pair) for pair in pairs)
def calculate_bottleneck_cost(graph, rssi, bottleneck_dummy):
    comm_cost_finder = functools.partial(find_comm_cost, graph,rssi)
    return min(comm_cost_finder(bottleneck_dummy.one),comm_cost_finder(bottleneck_dummy.two))
def add_bottleneck_dummies(problem, bottleneck_dummies, bottleneck_vars, cost_fun):
    cost_function = (cost_fun(i)* bottleneck_vars[i] for i in bottleneck_dummies)
    return cost_function
def add_cost_function(problem, dummy, dummy_vars, cost_calculator):
    cost_function = (cost_calculator(i)*dummy_vars[i] for i in dummy)
    return cost_function
def combine_costs(cost1, cost2):
    return itertools.chain(cost1, cost2)
def cost_adder(problem, costs):
    problem+=pulp.lpSum(costs)
    return problem
def find_communication_power(a, b, rssi):
    """a and b are the names of processor"""
    return rssi[a][b]
def find_computation_power(a, processors):
    return processors[a]
def find_comm_cost(graph, rssi, a_dummy):
    parent = a_dummy.parent
    child = a_dummy.child
    parent_node = graph[a_dummy.edge[0]]
    child_node = graph[a_dummy.edge[1]]
    child_node_name = a_dummy.edge[1]
    child_node_idx = int(child_node_name.split('_')[1])
    comm_size = parent_node['edge_w'].get(child_node_idx,0)
    try:
        comm_strength = rssi[parent].get(child)
        if not (comm_strength):
            comm_strength = rssi[child][parent]
        comm_cost_next = comm_size*comm_strength
    except Exception as e:
        msg = 'coudlnt calc cost,\n {},\n {}'.format(rssi.get(parent), rssi.get(child))
        raise(Exception(msg))
    #comm_cost_next = comm_size*rssi[parent].get(child,reverse)
    return comm_cost_next
def find_comp_cost(graph, processors, a_dummy):
    """comp cost for an edge is the computational load of the parent task
    scaled by the speed of the proposed parent node"""
    parent_node = graph[a_dummy.edge[0]]
    num_childers = len(parent_node['children'])
    parent = a_dummy.parent
    child_type = get_stage(a_dummy)
    speed = 1 if child_type == 'M' else processors[parent]
    comp_cost = (parent_node['node_w']/num_childers)/speed
    return comp_cost
def find_cost(graph, processors, rssi, a_dummy):
    """node is a key value pair, assignments is a named tuple"""
    comp = find_comp_cost(graph, processors, a_dummy)
    comm = find_comm_cost(graph, rssi, a_dummy)
    return comp+comm
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
                "sum of dummies equals 1 for: "+str(random.random()))
    return problem
###=======make sure edge assignments match at nodes======###
def match_parentchild(edge, edges):
    """find any neighbouring edges of node"""
    return ((edge,i) for i in edges if i[0] == edge[1])
def find_neighboring_edges(graph):
    """find all pairs of edges where child edge_i = parent edge_j"""
    edges = list(find_graph_edges(graph))
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
    matching_in_edge = [i for i in dummies if i.edge == in_edge]
    matching_out_edge = [i for i in dummies if i.edge == out_edge]
    return lazy_flattener((inconsistent_with_one(i, matching_out_edge)
                      for i in matching_in_edge))
def all_inconsistents(graph, dummies):
    """this function applies the find_inconsistent_assignments function
    over the whole graph: first by finding all pairs of in_edge, out_edge
    and then simply applying the function to each of these pairs in turn"""
    edge_pairs = list(find_neighboring_edges(graph))
    catcher = functools.partial(edgepair_inconsistents, dummies)
    wrong_nodes = lazy_flattener((catcher(*i) for i in edge_pairs))
    return wrong_nodes

def inout_consistency(graph, dummies, p, dummy_vars):
    all_matchers = list(all_inconsistents(graph, dummies))
    for inconsistent_pair in all_matchers:
        description = json.dumps(inconsistent_pair)
        added_dummy_vars = [dummy_vars[i] for i in inconsistent_pair]
        p += (pulp.lpSum(added_dummy_vars)<=1,
        "pick one of mutex pair: "+description)
    return p
def formulate_LP(graph, constraints, processors, rssi):
    d_gen = functools.partial(generate_dummies, graph, constraints)
    d = list(d_gen())
    problem = pulp.LpProblem("DAG",pulp.LpMinimize)
    dummy_vars = pulp.LpVariable.dicts("Sensor", d,0, 1, 'Binary')
    cost_calculator = functools.partial(find_cost, graph, processors, rssi)
    edge_costs = add_cost_function(problem, d, dummy_vars, cost_calculator)
    ###
    bottleneck_pairs = find_all_bottleneck_pairs(d)
    bottleneck_dummies = list(create_bottleneck_dummies(bottleneck_pairs))
    bottleneck_vars = pulp.LpVariable.dicts("Bottlenecks", bottleneck_dummies, 0, 1, 'Binary')
    bottleneck_cost_calculator = functools.partial(calculate_bottleneck_cost, graph, rssi)
    bottleneck_costs = add_bottleneck_dummies(problem, bottleneck_dummies, bottleneck_vars, bottleneck_cost_calculator)
    problem = cost_adder(problem, combine_costs(edge_costs, bottleneck_costs))
    #print('edge costs: ', edge_costs)
    #problem = cost_adder(problem, edge_costs)
    problem = constrain_bottlenecks(bottleneck_vars, dummy_vars, problem, bottleneck_dummies)
    problem = edge_uniqueness(problem, d, dummy_vars)
    problem = inout_consistency(graph, d, problem, dummy_vars)
    return problem
def solver(p):
    print('trying to solve')
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
    return sorted([(k,v) for k,v in node_assignment_pairs.items()], key=get_level)
def is_assignment(str):
    return str.split('(')[0]=="Sensor_Assignment"
def output(solution):
    all_nonzero = [(v.name,v.varValue) for v in solution.variables() if v.varValue >0]
    grouped = [list(g) for k,g in itertools.groupby(all_nonzero, keyfunct)]
    chosen = [max(i, key = get_val) for i in grouped]
    converted = [to_tuples(i[0]) for i in chosen if is_assignment(i[0])]
    node_assignment_pairs = tuples_to_node_assignment_pairs(converted)
    grouped_by_level = {k:list(g) for k,g in itertools.groupby(node_assignment_pairs, get_level)}
    chosen = {k: [i[1] for i in sortbynode(g)] for k,g in grouped_by_level.items()}
    return chosen, pulp.value(solution.objective)

solution_pipe = helper.pipe(formulate_LP, solver, output)
def tr(rssi):
    sub_dict = lambda d: {k:round(v,2) for k,v in d.items()}
    return {k:sub_dict(v) for k,v in rssi.items()}
def solve_DAG(code, rssi, processors, ack, bw=None):
    graph = dag_former.generate_weighted_graph(code)
    constraints = dag_former.generate_constraints(code, graph)
    if not bw:
        bw = bandwidth.get_full_bw(rssi,ack=ack)
    solution,time = solution_pipe(graph, constraints, processors, bw)
    return {'sol':solution,'graph':graph,'bw':tr(bw),'px':processors,'totaltime':time,'ack':ack}#add graph and bw for reconstruction

