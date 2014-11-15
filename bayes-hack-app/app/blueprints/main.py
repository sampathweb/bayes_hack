from flask import Blueprint, render_template
import pandas as pd
import mpld3
import seaborn as sns  # imported for Styling matplotlib plots
import json

main = Blueprint('main', __name__)


@main.route('/')
def index():
    df = pd.DataFrame({
        'year': [2010, 2011, 2012, 2013, 2014],
        'values': [1, 2, 3, 4, 5]
        })
    ax = df.plot(x='year', y='values', kind='bar', figsize=(12, 8))
    mpld3_data = mpld3.fig_to_dict(ax.get_figure())
    return render_template('index.html', mpld3_data=json.dumps(mpld3_data))
