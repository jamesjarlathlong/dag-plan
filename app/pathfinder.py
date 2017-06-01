import copy
import functools
def get_minweighted_unvisited_key(weights, unvisited_keys):
	"""from the weight dictionary return the key corresponding
	to the minimum weight"""
	candidate_weights = {k:v for k,v in weights.items() if k in unvisited_keys}
	return min(candidate_weights, key=lambda k: candidate_weights[k][0])

def adjust(adjust_function, kv, k):
    new_kv = copy.deepcopy(kv)
    new_kv[k] = adjust_function(new_kv[k])
    return new_kv

def relax_adjustment(weights, rootnode, current_cumulative_distance):
	tentative_distance = weights[rootnode][0] + current_cumulative_distance
	tentative_predecessor = rootnode
	adjustment_function = functools.partial(min, (tentative_distance, tentative_predecessor))
	return adjustment_function

def relax_one(node, weights, neighbor_kv):
	neighbor_weight = neighbor_kv[1]
	neighbor_name = neighbor_kv[0]
	adjuster = relax_adjustment(weights, node, neighbor_weight)
	return adjust(adjuster, weights, neighbor_name)

def relax_all_neighbors(current_node, weights, neighbor_weights):
	"""the pythonic way"""
	new_weights = weights
	for neighbor_name_weight in neighbor_weights:
		new_weights = relax_one(current_node, neighbor_name_weight, new_weights)
	return new_weights

def relax_the_neighbors(current_node, neighbor_distances, weights):
	relaxer = functools.partial(relax_one, current_node)
	return functools.reduce(relaxer, neighbor_distances, weights)

def weight_builder(source_node, graph):
	"""builds the initial weight data structure based on the graph and
	the source_node key, the sourcenode is initialised with weight 0
	while all others have infinite weight"""
	return {key: (0 if key==source_node else float("inf"), None) for key in graph}

def dijkstra_step(graph, weights, unvisited_keys):
	if not unvisited_keys:
		return weights
	else:
		current_node = get_minweighted_unvisited_key(weights, unvisited_keys)
		neighbors_of_current = [(k,v) for k,v in graph[current_node].items()]
		relaxed_weights = relax_the_neighbors(current_node, neighbors_of_current, weights)
		unvisited_keys.remove(current_node)
		return dijkstra_step(graph, relaxed_weights, unvisited_keys)
def run_dijkstra(source, graph):
	initial_weights = weight_builder(source, graph)
	initial_unvisited = [k for k in graph]
	dijkstra_stepper = functools.partial(dijkstra_step, graph)
	final_weight_tree = dijkstra_stepper(initial_weights, initial_unvisited)
	return final_weight_tree

def get_parent(node, tree):
	return tree[node][1]

def shortest_path_from_tree(tree, destination):
	parent = get_parent(destination, tree)
	if not parent:
		return [destination]
	else:
		return [destination] + shortest_path_from_tree(tree, parent)

def just_weights(tree):
	"""tree looks like  {'b': (17, 'd'), 'a': (0, None), 'e': (7, 'c'), 'd': (9, 'c'), 'c': (2, 'a')}
	we return just {'b':17, 'e':7...}"""
	return {k:v[0] for k,v in tree.items() if v[0]}

def connect_full(graph):
	return {k: just_weights(run_dijkstra(k, graph)) for k in graph}
	

