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

# These have some number of 
categorical_features = [
    u'_projectid', u'_teacher_acctid', u'_schoolid', u'school_ncesid',
    u'school_city', u'school_state', u'school_zip', u'school_metro',
    u'school_district', u'school_county',
    u'teacher_prefix', u'primary_focus_subject', u'primary_focus_area',
    u'secondary_focus_subject', u'secondary_focus_area', u'resource_type',
    u'poverty_level', u'grade_level',]

# All of these are tf
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
    return pd.read_csv(fn,
                       parse_dates=['date_expiration','date_thank_you_packet_mailed',
                                    'date_completed', 'date_posted'])
        
##############################
# Construct full data sets
##############################

def make_data_set(zz, state='CA'):
    ww = pre_screen(zz)
    # important to featureize _without_ restricting to state so that categorization happens correctly
    xx, enc, fwd, bkw = featureize(ww)
    # now select by state
    
    yy_all = funded_or_not(zz)
    # Pull out just the ones that are included 
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
    model = skl.ensemble.RandomForestRegressor()
    model.fit(xx,yy)
    return model

def make_ca_model(zz):
    xx, yy, idx, enc, fwd, bkw = make_data_set(zz, 'CA')
    print xx.shape
    model = skl.ensemble.RandomForestRegressor()
    model.fit(xx,yy)
    can(model, 'ca-model.pkl')
    return model

##############################
# Featurization
##############################

def cat2numerical(feature,nan_column=False):
    categories = set(feature)
    if not nan_column:
        categories = [cat for cat in categories if cat is not np.nan]
    print len(categories), categories
    new_features = []
    for cat in categories:
        col = pd.Series(np.zeros(len(feature)))
        col[feature == cat] = 1
        new_features.append(col)
    return pd.DataFrame(new_features).T

# decision tree for funded vs. not funded
def discretize(col):
    col = np.asarray(col)
    vals = sorted(list(set(col)))
    result = np.ones(len(col), 'i')
    result_dict = {}
    for idx, val in enumerate(vals):
        result[col==val] = idx
        result_dict[idx] = val
    return result, result_dict

def pre_screen(zz, state='CA'):
    by_state = zz[zz.school_state==state]
    return by_state[features].dropna(how='any').copy()
    
def featureize(zz):
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

    #features = ['num', 'city', 'color', 'val']
    #categorical_features = ['city', 'color']

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

