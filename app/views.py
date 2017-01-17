from app import app
from flask import (Flask, request, Response)
import json
@app.route('/solve', methods=['POST'])
def solve():
	code = request.data
	print('called: ', data)
	return Response(json.dumps({'res':4}), status = 200, )

def get_code_comp_cost(code):
	pass
def get code_comm_cost(code):
	pass