from flask import Blueprint, render_template
import pandas as pd
import mpld3
import seaborn as sns  # imported for Styling matplotlib plots
import json

import models
from map_plot import school_map
from funding_explore import funding_by_poverty_levels

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')


@main.route('/schools')
def schools():
    mpld3_data = mpld3.fig_to_dict(school_map()[0].get_figure())

    table_df = models.get_product_list(params={'item_name': '%books%'})
    table_html = table_df.head(20).to_html(classes=['table'])

    return render_template('schools.html', mpld3_data=json.dumps(mpld3_data), data_table=table_html)


@main.route('/funding-explore')
def funding_explore():
    # mpld3_data = mpld3.fig_to_dict(funding_by_poverty_levels().get_figure())

    # return render_template('funding_explore.html', mpld3_data=json.dumps(mpld3_data))
    return render_template('funding_completion.html')
