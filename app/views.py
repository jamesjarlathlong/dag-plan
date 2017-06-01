from app import app
#from models import Result
from flask import (Flask, request, Response)
import json
import functools
import app.dag_former as dag_former
import app.dag_solver as dag_solver
import app.bandwidth_calculator as bandwidth
@app.route('/solve', methods=['POST'])
def solve():
	code = request.form['code']
	solution = solve_LP(code)
	print('sol: ', solution)
	return Response(json.dumps({'res':solution}), status = 200)
def solve_LP(code):
	total_num = 10
	rssi = create_rssi(total_num)
	processors = create_processors(total_num)
	solution = dag_solver.solve_DAG(code, rssi, processors)
	return solution
def create_processors(total_num):
    return {0:1, 1:0.01, 2:0.01,3:0.01,4:0.01,5:0.01}
def create_rssi(total_num):
    def to_others(total,i):
        def rssi(i,j):
        	if i!=j:
        		return -50
        return{j:rssi(i,j) for j in range(total)}
    other_gen = functools.partial(to_others,total_num)
    return{i:other_gen(i) for i in range(total_num)}