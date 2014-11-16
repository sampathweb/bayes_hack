from flask import Blueprint, render_template
import json

product = Blueprint('product', __name__)




@product.route('/product')
def index():
	products = [
		{"value":"prod1", "label":"Product 1"},
		{"value":"prod2", "label":"Product 2"},
		{"value":"prod3", "label":"Product 3"},
		{"value":"prod4", "label":"Product 4"}
	]
	return render_template('product.html', productlist=products)
