from flask import Blueprint, render_template, g
import json
import pandas as pd

product = Blueprint('product', __name__)




@product.route('/product')
def index():
	states = [
		{"value":"NY", "label":"New York"},
		{"value":"CA", "label":"California"},
	]
	#data = pd.read_sql('select count(*) as c from donorschoose_projects', g.db_engine)
	localization = pd.read_sql('select distinct school_county from donorschoose_projects', g.db_engine)

	return render_template('product.html', localization=localization.to_json()) #, count=data['c'].iloc[0])
