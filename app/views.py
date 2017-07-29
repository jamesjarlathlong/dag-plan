from app import app
#from models import Result
from flask import (Flask, request, Response)
import json
import functools
import app.dag_former as dag_former
import app.dag_solver as dag_solver
import app.bandwidth_calculator as bandwidth
import app.node_emulator as node_emulator
@app.route('/solve', methods=['POST'])
def solve():
	print('got a req',request,request.form)
	code = request.form['code']
	solution = solve_LP(code)
	print('sol: ', solution)
	return Response(json.dumps({'res':solution}), status = 200)
def solve_LP(code, rssi=None, proc=None):
	graph = dag_former.generate_weighted_graph(code)
	constraints = dag_former.generate_constraints(code, graph)
	total_num = 10
	if not rssi:
		rssi = create_rssi(total_num)
	if not proc:
		proc = create_processors(total_num)
	print(rssi)
	solution = dag_solver.solution_pipe(graph, constraints, proc, rssi)
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

def create_processors(total_num):
    return {k:1 if k==0 else 0.05 for k in range(total_num)}
def create_rssi(total_num):
    def to_others(total,i):
        def rssi(i,j):
        	if i!=j:
        		return -50
        return{j:rssi(i,j) for j in range(total)}
    other_gen = functools.partial(to_others,total_num)
    return{i:other_gen(i) for i in range(total_num)}
