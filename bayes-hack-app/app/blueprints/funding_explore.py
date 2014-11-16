from flask import g
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def funding_by_poverty_levels():

    projects = pd.read_sql('''
        select poverty_level, date_posted, date_completed
        from donorschoose_projects
        where date_completed > "2005-01-01"
            ''', \
        g.db_engine \
    )

    # number of dates to completion -- only look at the completed projects
    #projects_ca = projects.ix[projects.school_state == 'CA']
    date2completion = pd.to_datetime(projects['date_completed']) - pd.to_datetime(projects['date_posted'])

    # number of days to completion -- converted into integer
    # probably non optimal but why can't i get apply to work
    def convert_date(d):
        if pd.isnull(d):
            return None
        else:
            return d.days
    # date2completion.apply(lambda x : convert_date(x))
    date2completion2 = np.zeros(len(date2completion))
    for i,d in enumerate(date2completion):
        date2completion2[i] = convert_date(d)
    date2completion2 = pd.Series(date2completion2)

    # bin into categories
    bins = [0, 1., 30., 150., date2completion2.max()]
    indices = ['1 day', '30 days', '150 days', '>150 days']
    cutsbydays = pd.cut(date2completion2, bins, include_lowest=True,labels=indices)

    ax = projects['poverty_level'].groupby(cutsbydays).value_counts().unstack(level=0).plot(kind='barh')
    return ax
