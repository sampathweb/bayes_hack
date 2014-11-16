from flask import Blueprint, render_template
import json

product = Blueprint('product', __name__)




@product.route('/product')
def index():
	states = [
		{"value":"NY", "label":"New York"},
		{"value":"CA", "label":"California"},
	]
	return render_template('product.html', stateslist=states)
