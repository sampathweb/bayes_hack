#! ../env/bin/python
from flask import Flask, render_template, redirect, url_for, request, send_from_directory, g

import matplotlib
matplotlib.use('agg')


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
            pass
            # g.db_engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

    # register our blueprints
    from app.blueprints import main, product
    app.register_blueprint(main)
    app.register_blueprint(product)

    
    return app
