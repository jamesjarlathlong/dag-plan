from app import db
from sqlalchemy.dialects.postgresql import JSON


class Computation(db.Model):
    __tablename__ = 'computation'

    id = db.Column(db.Integer, primary_key=True)
    node_id = db.Column(db.Integer)
    server_time = db.Column(db.Float)
    node_time = db.Column(db.Float)
    date = db.Column(db.DateTime)

    def __init__(self, server_time, node_time, date):#
        self.node_id = node_id
        self.server_time = server_time
        self.node_time = node_time
        self.date = date

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Communication(db.Model):
    __tablename__ = 'communication'

    id = db.Column(db.Integer, primary_key=True)
    node_id = db.Column(db.Integer)
    to_node_id = db.Column(db.Integer)
    data_length = db.Column(db.Integer)
    comm_time = db.Column(db.Float)
    date = db.Column(db.DateTime)

    def __init__(self, server_time, node_time, date):#
        self.node_id = node_id
        self.to_node_id = to_node_id
        self.data_length = data_length
        self.comm_time = comm_time
        self.date = date

    def __repr__(self):
        return '<id {}>'.format(self.id)