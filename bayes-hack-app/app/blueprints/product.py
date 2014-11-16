from flask import Blueprint, render_template, g, request, jsonify
import json
import mysql.connector
from random import randint
import mpld3
import seaborn as sns  # imported for Styling matplotlib plots
import json
import numpy as np
from pred_model import prob_plot

db = mysql.connector.connect(user='root', password='bayeshack',host='bayesimpact.soumet.com', database='bayes')

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
    school_state = ["NY", "CA","TX"]

    # school_state = ["NY","NC","IL","CA","TX","SC","LA","OK","MS","AL","NE","MD","NJ","MO","IN","MA","VA","CO",
    #                     "KS","CT","MI","WA","RI","KY","AZ","MN","PA","DC","UT","OH","GA","ME","NM","FL","DE",
    #                     "OR","TN","WV","ID","NV","IA","HI","WI","MT","VT","NH","AR","SD","AK","ND","WY"]
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

    bool_fields = ['school_magnet', 'school_year_round', 'school_charter', 'teacher_teach_for_america', 'school_kipp', \
'teacher_ny_teaching_fellow', 'school_nlns', 'school_charter_ready_promise']

    # This is the code to make an example
    pred_data = np.zeros(g.model_arr_size)
    for col in data:
        idx = g.model_fwd_xform(col, data[col])
        if col in bool_fields:
            if data[col] == 'true':
                pred_data[idx] = 1
            else:
                pred_data[idx] = 0
        elif col == 'total_price_excluding_optional_support':
            pred_data[idx] = float(data[col])
    if data['school_state'] == 'NY':
        pred_prob = g.ny_model.predict(pred_data)
    elif data['school_state'] == 'CA':
        pred_prob = g.ca_model.predict(pred_data)
    else:
        pred_prob = g.tx_model.predict(pred_data)

    print pred_data
    print pred_prob
    # mpld3_data = mpld3.fig_to_dict(prob_plot(g.xx).get_figure())

    return jsonify({"res":int(pred_prob * 100), "input":data})
    # , mpld3_data: json.dumps(mpld3_data)})
