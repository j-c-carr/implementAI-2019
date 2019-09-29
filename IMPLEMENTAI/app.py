from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin


#init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/dummy": {"origins": "http://localhost:5000"}})





#Database
app.config['SQLALCHEMY_DATABASE_URI']= 'sqlite:///'+os.path.join(basedir, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
#Init db
db = SQLAlchemy(app)
#init ma
ma = Marshmallow(app)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    description = db.Column (db.String(200))
    price = db.Column(db.Float)
    qty = db.Column(db.Integer)

    def __init__(self, name, description, price, qty):
        self.name = name
        self.description = description
        self.price = price
        self.qty = qty


class ProductSchema(ma.Schema):
    class Meta:
        fields =  ('id','name','description','price','qty')


#init schema
product_schema = ProductSchema()
products_schema = ProductSchema(many=True)


#create a product
@app.route('/product', methods=['POST'])
def add_product():
    name = request.json['name']
    description = request.json['description']
    price = request.json['price']
    quantity = request.json['qty']

    new_product = Product(name,description,price,qty)
    db.session.add(new_product)
    db.session.commit()

    return product_schema.jsonify(new_product)

#get all products
@app.route('/product', methods=['GET'])
def get_all():
    all_products = Product.query.all()
    #equivalent of SELECT * FROM TABLE

    result = products_schema.dump(all_products)
    return jsonify(result.data)

@app.route('/dummy', methods=['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def get_dummy():
    dummy = "Hellooooooo world"
    return dummy


#Run Server
if __name__ == '__main__':
    app.run(debug=True)
