from flask import g
import pandas as pd


def get_product_list(params):
    df = pd.read_sql('''
        select item_name
        from donorschoose_resources
        where item_name like %(item_name)s
        limit 100
            ''', \
        g.db_engine, \
        params={'item_name': params['item_name']}
    )
    return df


