from flask import g
import pandas as pd


def get_product_list(params):
    df = pd.read_sql('''
        select item_name
        from donorschoose_resources
        limit 100
            ''', \
        g.db_engine \
    )
    return df
