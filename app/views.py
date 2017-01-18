from app import app
from flask import (Flask, request, Response)
import json
import app.dag as dag
print([i for i in dag.generate_graph('a')])
@app.route('/solve', methods=['POST'])
def solve():
	code = request.data
	print('called: ', data)
	return Response(json.dumps({'res':4}), status = 200, )