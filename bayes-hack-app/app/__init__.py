#! ../env/bin/python
from flask import Flask, render_template, redirect, url_for, request, send_from_directory, g
from sqlalchemy import create_engine

import matplotlib
matplotlib.use('agg')

from app.blueprints import pred_model


def load_model():
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:bayeshack@bayesimpact.soumet.com/bayes'
    db_engine = create_engine(SQLALCHEMY_DATABASE_URI)
    zz = pred_model.project_data(db_engine)
    xx, yy, idx, enc, fwd, bkw = pred_model.make_data_set(zz)
    # ny_model = pred_model.uncan('ny-model.pkl')
    ca_model = pred_model.uncan('ca-model.pkl')
    # tx_model = pred_model.uncan('tx-model.pkl')
    print "***", xx.shape
    return zz, xx, yy, xx.shape[1], fwd, ny_model, ca_model, tx_model


# ny_model.predict(array)

model_zz, model_xx, model_yy, model_arr_size, model_fwd_xform, ny_model, ca_model, tx_model = load_model()


def create_app(object_name, env):
    app = Flask(__name__)

    app.config.from_object(object_name)

    @app.errorhandler(500)
    def error_handler_500(e):
        if app.config['APP_SERVER'] != 'DEV':
            return render_template('500.html'), 500

    @app.errorhandler(404)
    def error_handler_404(e):
        if app.config['APP_SERVER'] != 'DEV':
            return render_template('404.html'), 404

    @app.route('/robots.txt')
    @app.route('/sitemap.xml')
    def static_from_root():
        return send_from_directory(app.static_folder, request.path[1:])

    @app.route('/favicon.ico')
    def favicon():
        return redirect(url_for('static', filename='img/favicon.ico'))

    @app.before_request
    def before_request():
        g.app_server = app.config['APP_SERVER']
        if 'db_engine' not in 'g':
            g.db_engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
        if 'model_zz' not in 'g':
            g.model_zz = model_zz
            g.model_xx = model_xx
            g.model_yy = model_yy
            g.model_arr_size = model_arr_size
            g.model_fwd_xform = model_fwd_xform
            g.ny_model = ny_model
            g.tx_model = tx_model
            g.ca_model = ca_model

    # register our blueprints
    from app.blueprints import main, product
    app.register_blueprint(main)
    app.register_blueprint(product)

    return app
