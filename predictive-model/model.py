###
# What needs to be done?
# - Better featurization function -- Don't require web server to load all the data
# - Better model -- logistic regression?
#

###
# Feature importances spat out by the random forest model
# 
# total_price_excluding_optional_support | |  0.51
# grade_level |  Grades PreK-2 |  0.022
# grade_level | Grades 3-5 | 0.022
# school_magnet |  | 0.021
# poverty_level | highest poverty | 0.019
# school_charter | | 0.019
# poverty_level | high poverty | 0.019
# grade_level | Grades 6-8 | 0.018
# teacher_prefix | Ms. | 0.017
# grade_level | Grades 9-12 | 0.016

import cPickle
import sklearn.ensemble
import sklearn as skl
import numpy as np
import pandas as pd
import pylab as pl

##############################
# Which features to include,
# what are their types?
##############################

# All of the features available in opendata_projects.csv
# The ones we're using are uncommented.
features = [
    # categorical, lots of categories
    # u'_projectid', u'_teacher_acctid', u'_schoolid', u'school_ncesid',
    ### u'school_latitude', u'school_longitude',        
    # categorical, lots of categories
    # u'school_city',
    u'school_state',
    # categorical, lots of categories
    # u'school_zip',
    u'school_metro',
    # categorical, lots of categories
    # u'school_district', u'school_county',
    # Categorical, few categories, use
    u'school_charter', u'school_magnet', u'school_year_round', u'school_nlns', u'school_kipp', u'school_charter_ready_promise', u'teacher_prefix', u'teacher_teach_for_america', u'teacher_ny_teaching_fellow', u'primary_focus_subject', u'primary_focus_area',
    ### u'secondary_focus_subject', u'secondary_focus_area',
    u'resource_type',
    u'poverty_level',
    u'grade_level',
    # Screwball Numerical, mostly missing
    #u'vendor_shipping_charges', u'sales_tax', u'payment_processing_charges', u'fulfillment_labor_materials',
    # Numerical
    u'total_price_excluding_optional_support', # u'total_price_including_optional_support',
    ### u'students_reached', u'total_donations', u'num_donors', u'eligible_double_your_impact_match', u'eligible_almost_home_match',
    # Exclude, giveaway
    # u'funding_status', u'date_posted', u'date_completed', u'date_thank_you_packet_mailed', u'date_expiration'
]

# Categorical features
categorical_features = [
    u'_projectid', u'_teacher_acctid', u'_schoolid', u'school_ncesid',
    u'school_city', u'school_state', u'school_zip', u'school_metro',
    u'school_district', u'school_county',
    u'teacher_prefix', u'primary_focus_subject', u'primary_focus_area',
    u'secondary_focus_subject', u'secondary_focus_area', u'resource_type',
    u'poverty_level', u'grade_level',]

# True/false binary features
binary_features = [
    u'school_charter', 
    u'school_magnet',
    u'school_year_round',
    u'school_nlns',
    u'school_kipp',
    u'school_charter_ready_promise',
    u'teacher_teach_for_america',
    u'teacher_ny_teaching_fellow',
    u'eligible_double_your_impact_match',
    u'eligible_almost_home_match']

##############################
# I/O
##############################

def project_data(fn='../../data/opendata_projects.csv'):
    """Read projects.csv into a Pandas dataframe"""
    return pd.read_csv(fn,
                       parse_dates=['date_expiration','date_thank_you_packet_mailed',
                                    'date_completed', 'date_posted'])
        
##############################
# Construct full data sets
##############################

def make_data_set(zz):
    """Convert dataframe to arrays suitable for scikit learn"""
    ww = pre_screen(zz)
    xx, enc, fwd, bkw = featureize(ww)
    
    yy_all = funded_or_not(zz)
    # Pull out just the ones that are included in the inputs
    yy = yy_all.loc[ww.index]
    return xx.toarray(), yy.values.astype('i'), ww.index, enc, fwd, bkw

def funded_or_not(zz):
    return pd.isnull(zz.date_completed-zz.date_posted)

def time_to_fund():
    return pd.isnull(zz.date_completed-zz.date_posted)

##############################
# Model Fitting
##############################

def fit_model(xx,yy):
    """Fit model given inputs xx and outputs yy"""
    model = skl.ensemble.RandomForestRegressor()
    model.fit(xx,yy)
    return model

def make_state_model(zz, state='CA'):
    """Make and save model for California data for the web site."""
    xx, yy, idx, enc, fwd, bkw = make_data_set(zz)

    # Pull out the entries for the state just before fitting the model
    col = fwd('school_state', state)
    keep = np.nonzero(xx[:,col])
    xx, yy, idx = xx[keep], yy[keep], idx[keep]

    model = skl.ensemble.RandomForestRegressor()
    model.fit(xx,yy)
    can(model, 'model-%s.pkl' % state)
    return model

##############################
# Featurization
##############################

def pre_screen(zz):
    """Apply filters to data e.g. no NaNs, by state, etc"""
    return zz[features].dropna(how='any').copy()
    
def featureize(zz):
    """Convert dataframe into arrays suitable for scikit learn

    Input: dataframe from opendata_projects.csv

    Output: xx, encode, forward_transform, reverse_transform

    xx: a N_examples x N_features array suitable for use as input data
    for scikit learn fit() functions

    encode: a function that takes a dataframe and produces xx

    forward_transform: a function that takes a feature name and (if
    applicable) a category name and returns the column of xx that
    corresponds to that feature.  Ex:

    forward_transform('school_longitude') => 55
    forward_transform('school_state', 'CA') => 3
    forward_transform('school_state', 'NY') => 17

    so x[:,3] will be nonzero if the school is in California

    reverse_transform: a function that takes a column of xx and maps
    it to a feature name and (if applicable) category name.  Ex:

    reverse_transform(55) => ('school_longitude', None)
    reverse_transform(3) => ('school_state', 'CA')
    reverse_transform(17) => ('school_state', 'NY')

    Feature names are the column names in the data frame.  Category
    names are taken from the unique set of values from a given
    categorical column.

    """

    def partial_encode(xx):
        result = []
        for ff in features:            
            aa = xx[ff].values
            if ff in binary_features:
                result.append(np.where(aa=='t', 1, 0))
            elif ff in categorical_features:                
                result.append(label_encoder[ff].transform(aa))
            else:
                result.append(aa)
        return np.array(result).transpose()

    def encode(xx):        
        return enc.transform(partial_encode(xx))
    
    # define a function that does the forward mapping
    # given ff, lab (might be none)
    def forward_transform(ff, label=None):
        if ff in categorical_features:
            assert label is not None
            idx = cat_labels.index(ff)
            base = enc.feature_indices_[idx]
            offset = label_encoder[ff].transform(label)
        else:
            base = enc.feature_indices_[-1]
            offset = non_cat_labels.index(ff)
        return base + offset

    # define a function that does the reverse mapping
    def reverse_transform(col):
        assert col >=0
        if col < enc.feature_indices_[-1]:
            # categorical
            idx = enc.feature_indices_.searchsorted(col,side='right')-1
            base = enc.feature_indices_[idx]
            offset = col - base 
            feature = cat_labels[idx]
            label = label_encoder[feature].inverse_transform(offset)
            return feature, label
        else:
            offset = col - enc.feature_indices_[-1]
            return non_cat_labels[offset], None
        return key, label

    # # Small arrays of features / categorical features, used for testing.
    # features = ['num', 'city', 'color', 'val']
    # categorical_features = ['city', 'color']

    # Take the desired features and drop any rows with missing values
    #zz = zz[features].dropna(how='any')
    
    label_encoder = {}
    non_cat_labels = []
    cat_labels = []
    cat_feature = []
    for ff in features:            
        if ff in categorical_features:
            label_encoder[ff] = skl.preprocessing.LabelEncoder()
            label_encoder[ff].fit(zz[ff].values)
            cat_labels.append(ff)
            cat_feature.append(True)
        else:
            non_cat_labels.append(ff)
            cat_feature.append(False)

    enc = skl.preprocessing.OneHotEncoder(categorical_features=cat_feature)    
    xx_partial = partial_encode(zz)
    enc.fit(xx_partial)
    xx = enc.transform(xx_partial)
        
    return xx, encode, forward_transform, reverse_transform

##############################
# Utilities
##############################

def can(obj, file, protocol=2):
    """More convenient syntax for pickle, intended for interactive use

    Most likely:
    >>> can([1,2,3], 'file.dat')
    But can also do:
    >>> with open('file.dat', 'w') as f: can([1,2,3], f); can((3,4,5), f)

    """
    if type(file) is str: f=open(file,'wb')
    else: f=file

    cPickle.dump(obj, f, protocol=protocol)

    if type(file) is str: f.close()

def uncan(file):
    """More convenient syntax for pickle, intended for interactive use

    Most likely:
    >>> obj = uncan('file.dat')
    But can also do:
    >>> with open('file.dat') as f: foo = uncan(f); bar = uncan(f)

    """
    # If filename, should this read until all exhausted?
    if type(file) is str: f=open(file, 'rb')
    else: f=file    

    obj = cPickle.load(f)

    if type(file) is str: f.close()

    return obj

