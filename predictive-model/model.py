###
# What needs to be done?
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
import sklearn.linear_model
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
# Example code to fit and use a model
##############################

def make_model_from_scratch():
    zz = project_data()
    xx, yy, idx, encoder = make_data_set(zz)
    model, encoder = make_model(xx, yy, idx, encoder, state='CA'):

    if state is not None:
        filename = '%s-%s.pkl' % (prefix, state)
    else:
        filename = '%s.pkl' % prefix

    # Could use the same encoder for multiple models, but reduce the
    # possibility of screw-ups, always store the model and encoder
    # together.
    can((model, encoder), filename)

def use_model(state='CA'):
    """Example code to load and use a state model"""
    # This happens _once_ when the web server is started
    if state is not None:
        filename = '%s-%s.pkl' % (prefix, state)
    else:
        filename = '%s.pkl' % prefix
    model, encoder = uncan(filename)

    # This happens when the web server responds to a request
    request = dict(school_state='CA',
                   school_charter=True,
                   total_price_excluding_optional_support=11.1)
    vv = encoder.encode_dict(request)
    return model.predict(vv)

##############################
# Featurization
##############################

class DonorsChooseEncoder(object):
    """Object that transforms between dataframes/dicts and feature vectors

    The web server should be able to unpickle this object and use it
    to populate the feature vector _without_ loading all the data.  In
    particular the web server should be able to correctly map feature
    names to entries in the feature vector based on the features, etc,
    that were used _when that particular model was trained_.

    Therefore when a model is trained, one should pickle both the
    model and this feature encoder.  Then if we train a model, and
    then update this source file to include/exclude more features, it
    _will not_ mess up the feature mapping and make the model return
    nonsense.

    """

    def __init__(self, df):
        """Make a feature encoder based on a dataframe

        Given a list of features, categorical features, and binary
        (true/false) features, make a set of feature vectors suitable
        for ingestion into scikit learn.  Ie, categorical features are
        exploded into n-dimensional vectors, one dimension for each
        label, etc.

        Feature names are the column names in the data frame.  Category
        names are taken from the unique set of values from a given
        categorical column.

        """
        # Capture module variables
        self._features = features
        self._categorical_features = categorical_features
        self._binary_features = binary_features

        # and construct the encoding
        self._make_encoder(df)

    def _make_encoder(self, df):
        """Convert dataframe into arrays suitable for scikit learn"""

        label_encoder = {}
        non_cat_labels = []
        cat_labels = []
        cat_feature = []

        for ff in self._features:
            if ff in self._categorical_features:
                label_encoder[ff] = skl.preprocessing.LabelEncoder()
                label_encoder[ff].fit(df[ff].values)
                cat_labels.append(ff)
                cat_feature.append(True)
            else:
                non_cat_labels.append(ff)
                cat_feature.append(False)

        encoder = skl.preprocessing.OneHotEncoder(categorical_features=cat_feature)

        # Save what we need to save
        self._encoder = encoder
        self._label_encoder = label_encoder
        self._cat_labels = cat_labels
        self._non_cat_labels = non_cat_labels

        # And do the final training of the encoder
        xx_partial = self._partially_encode_dataframe(df)
        encoder.fit(xx_partial)
        xx = encoder.transform(xx_partial)
        self.n_features = xx.shape[1]

    def reverse_transform(self, col):
        """Turn a column index into a feature/category name

        reverse_transform(55) => ('school_longitude', None)
        reverse_transform(3) => ('school_state', 'CA')
        reverse_transform(17) => ('school_state', 'NY')

        Feature names are the column names in the data frame.
        Category names are taken from the unique set of values from a
        given categorical column.

        """

        assert col >=0
        if col < self._encoder.feature_indices_[-1]:
            # categorical
            idx = self._encoder.feature_indices_.searchsorted(col,side='right')-1
            base = self._encoder.feature_indices_[idx]
            offset = col - base 
            feature = self._cat_labels[idx]
            label = self._label_encoder[feature].inverse_transform(offset)
            return feature, label
        else:
            offset = col - self._encoder.feature_indices_[-1]
            return self._non_cat_labels[offset], None
        return key, label

    def forward_transform(self, ff, label=None):
        """Turn feature/category names into a column index

        forward_transform('school_longitude') => 55
        forward_transform('school_state', 'CA') => 3
        forward_transform('school_state', 'NY') => 17

        so in the feature matrix x[:,3] will be nonzero if the school
        is in California

        """

        if ff in self._categorical_features:
            assert label is not None
            idx = self._cat_labels.index(ff)
            base = self._encoder.feature_indices_[idx]
            offset = self._label_encoder[ff].transform(label)
        else:
            base = self._encoder.feature_indices_[-1]
            offset = self._non_cat_labels.index(ff)
        return base + offset

    def encode_dataframe(self, df):
        """Turn a dataframe into a feature vector"""
        return self._encoder.transform(self._partially_encode_dataframe(df))

    def _partially_encode_dataframe(self,xx):
        """Encode text labels into numerical values"""
        result = []
        for ff in self._features:
            aa = xx[ff].values
            if ff in self._binary_features:
                result.append(np.where(aa=='t', 1, 0))
            elif ff in self._categorical_features:
                result.append(self._label_encoder[ff].transform(aa))
            else:
                result.append(aa)
        return np.array(result).transpose()

    def encode_dict(self, dd):
        """Turn a dict into a feature vector

        This does _no_ error checking (ie, ensuring that _some_ value
        is specified for each feature

        """
        vv = np.zeros(self.n_features)
        for kk in dd.keys():
            idx = self.forward_transform(kk, dd[kk])
            if kk in self._binary_features:
                if dd[kk]:
                    vv[idx] = 1.0
                else:
                    vv[idx] = 0.0
            elif kk in self._categorical_features:
                vv[idx] = 1.0
            else:
                vv[idx] = dd[kk]
        return vv

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
    # Don't seem to actually need pre-screening that removes entries
    # with NaNs.  Leave the logic to get the input and output vectors
    # to line up correctly, though, in case it's needed down the line.
    #
    # ww = pre_screen(zz)
    ww = zz
    encoder = DonorsChooseEncoder(zz)
    xx = encoder.encode_dataframe(ww)
    
    yy_all = funded_or_not(zz)
    # Pull out just the ones that are included in the inputs
    yy = yy_all.loc[ww.index]
    return xx.toarray(), yy.values.astype('i'), ww.index, encoder

def funded_or_not(zz):
    return pd.isnull(zz.date_completed-zz.date_posted)

def time_to_fund():
    return pd.isnull(zz.date_completed-zz.date_posted)

def pre_screen(zz):
    """Apply filters to data e.g. no NaNs, by state, etc"""
    return zz[features].dropna(how='any').copy()

##############################
# Model Fitting
##############################

def fit_model(xx,yy):
    """Fit model given inputs xx and outputs yy"""
    model = skl.linear_model.LogisticRegression()
    #model = skl.ensemble.RandomForestRegressor()
    model.fit(xx,yy)
    return model

def make_model(xx, yy, idx, encoder, state=None, prefix='model'):
    """Make and save model, possibly for a single state"""

    # Pull out the entries for the state just before fitting the model
    if state is not None:
        col = encoder.forward_transform('school_state', state)
        keep = np.nonzero(xx[:,col])
        xx, yy, idx = xx[keep], yy[keep], idx[keep]

    model = fit_model(xx,yy)
    return model, encoder

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

