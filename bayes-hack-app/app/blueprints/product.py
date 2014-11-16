from flask import Blueprint, render_template, g, request, jsonify 
import json
import mysql.connector
from random import randint

db = mysql.connector.connect(user='root', password='bayeshack',host='127.0.0.1', database='bayes')
	
product = Blueprint('product', __name__)


#@product.route('/', methods=['POST'])
@product.route('/get_products')
def get_products():
	cursor = db.cursor(True)
	query = ("SELECT item, item_name, item_unit_price, item_quantity from items_index where item_name like '%s%%' LIMIT 8;") % request.args.get('item_name')
	
	cursor.execute(query)
	results = { "items" : []}
	for item, item_name, item_unit_price, item_quantity in cursor:
		#print item_unit_price, item_name, item_quantity
		try:
			item_unit_price = float(item_unit_price)
			item_quantity = int(item_quantity)
		except:
			item_unit_price = 0
			item_quantity = 1

		results["items"].append(
			{"item_prefix":str(item.encode('ascii', 'ignore')), 
				"item_name":str(item_name.encode('ascii', 'ignore')),
				#"item_number":str(item_number.encode('ascii', 'ignore')),
				"item_unit_price":float(item_unit_price),
				"item_quantity":int(item_quantity)
			})
	cursor.close()
	#db.close()
	return jsonify(results)
	#items_match = pd.read_sql(sql, g.db_engine)
	#if request.method == 'POST':
	#return items_match.to_json()

@product.route('/product')
def index():
	states = [
		{"value":"NY", "label":"New York"},
		{"value":"CA", "label":"California"},
	]
	school_state = ["NY","NC","IL","CA","TX","SC","LA","OK","MS","AL","NE","MD","NJ","MO","IN","MA","VA","CO",
						"KS","CT","MI","WA","RI","KY","AZ","MN","PA","DC","UT","OH","GA","ME","NM","FL","DE",
						"OR","TN","WV","ID","NV","IA","HI","WI","MT","VT","NH","AR","SD","AK","ND","WY"]
	school_metro = [{"name":"Urban", "value":"urban"},
					{"name":"Rural", "value":"rural"},
					{"name":"Suburban", "value":"suburban"}]
	teacher_prefix = ["Mr.","Ms.","Mrs.","Dr.","Mr. & Mrs."]
	primary_focus_subject = ["Literacy","Health & Life Science","Performing Arts",
		"College & Career Prep","Character Education","Literature & Writing",
		"History & Geography","Environmental Science","Extracurricular",
		"Visual Arts","Early Development","Mathematics","Other","Applied Sciences",
		"Foreign Languages","Parent Involvement","Music","Economics","Sports",
		"Civics & Government","Social Sciences","Special Needs","Community Service",
		"ESL","Gym & Fitness","Health & Wellness","Nutrition"
	]
	primary_focus_area = ["Literacy & Language","Math & Science","Music & The Arts","Applied Learning","History & Civics","Health & Sports","Special Needs"]
	grade_level = ["Grades 9-12","Grades 6-8","Grades PreK-2","Grades 3-5"]
	resource_type = ["Books","Technology","Trips","Visitors","Other","Supplies"]
	poverty_level = [ {"name":"Highest Poverty", "value": "highest poverty"},
						{"name":"High Poverty", "value": "high poverty"},
						{"name":"Moderate Poverty", "value": "moderate poverty"},
						{"name":"Low Poverty", "value": "low poverty"}
	]
	#data = pd.read_sql('select count(*) as c from donorschoose_projects', g.db_engine)
	#localization = pd.read_sql('select distinct school_county from donorschoose_projects', g.db_engine)
	#localization = {"test":True}
	return render_template('product.html', 
				school_state=school_state, 
				school_metro=school_metro, 
				teacher_prefix=teacher_prefix,
				primary_focus_subject=primary_focus_subject,
				primary_focus_area=primary_focus_area,
				grade_level=grade_level,
				resource_type=resource_type,
				poverty_level=poverty_level)#, localization=localization.to_json()) #, count=data['c'].iloc[0])


@product.route('/check_prediction', methods=['POST'])
def check_prediction():
	print request.form.get("parameters")
	data = request.form
	#print data
	#print data['school_state']
	return jsonify({"res":randint(1,100), "input":data})