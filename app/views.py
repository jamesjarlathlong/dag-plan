from app import app
from flask import (Flask, request, Response)
import json
import functools
import app.dag_former as dag_former
import app.dag_solver as dag_solver

@app.route('/solve', methods=['POST'])
def solve():
	code = request.json.code
	solution = solve_LP(code)
	return Response(json.dumps({'res':solution}), status = 200)
def solve_LP(code):
	graph = dag_former.generate_weighted_graph(code)
	print(graph)
	constraints = dag_former.generate_constraints(code, graph)
	total_num = 10
	rssi = create_rssi(total_num)
	processors = create_processors(total_num)
	solution = dag_solver.solution_pipe(graph, constraints, processors, rssi)
	return solution
def create_processors(total_num):
    def processor_power(num):
        if num==0:
            return 10**5
        else:
            return 2*num
    return {i:processor_power(i) for i in range(total_num)}
def create_rssi(total_num):
    def to_others(total,i):
        def rssi(i,j):
            if i==j:
                return 10**6
            else:
                return 10
        return{j:rssi(i,j) for j in range(total)}
    other_gen = functools.partial(to_others,total_num)
    return{i:other_gen(i) for i in range(total_num)}
sol = solve_LP('a')
print('sol: ', sol)