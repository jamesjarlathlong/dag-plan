"""given rssi values converts to number of seconds taken for one message on average"""
import math
import functools
import app.helper as helper
import app.pathfinder as pathfinder
def probability_success(failure_when_perfect, rssi):
    """failure_when_perfect is the probability of a failed packet
    when the rssi is of very high quality
    rssi is in -db, usually in the range -40 to -100"""
    sigmoid = (1-failure_when_perfect)/(1+math.exp(-0.2*(70+rssi)))
    return sigmoid
our_ps = functools.partial(probability_success, 0.01)

def num_tries(prob_success):
	"""on average how many tries required per msg
	e.g. if prob_success is 100%-one try, if it's 10% -  10 tries"""
	return 1./prob_success
def time_per_msg(time_per_try,num_tries):
	return num_tries*time_per_try
def guess_time_per():
	throttle = 50 #ms
	baud = 9600 #/s
	bits_per_message = 100*8
	write_time = 1000*bits_per_message/baud #convert to ms by 1000*
	return throttle+write_time
eg_time_per_msg = functools.partial(time_per_msg, guess_time_per())
def time_per_byte(t_per_msg):
	byte_per_msg = 75
	msg_per_byte = 1./75
	return t_per_msg*msg_per_byte
rssi_to_time = helper.pipe(our_ps, num_tries, eg_time_per_msg, time_per_byte)
def one_node_to_time(single_rssi_dict):
	return {k: rssi_to_time(v) for k,v in single_rssi_dict.items() if v}
def rssi_graph_to_time(rssi_graph):
	print('rssi graph: ', rssi_graph)
	return {k: one_node_to_time(v) for k,v in rssi_graph.items()}

def dummy_self(k, node_dict):
	node_dict[k] = 1e-6
	return node_dict
def dummy_selves(time_graph):
	return {k: dummy_self(k,v) for k,v in time_graph.items()}
get_full_bw = helper.pipe(rssi_graph_to_time, pathfinder.connect_full, dummy_selves)

###to do -  fix bottleneck issue, implement connect full.