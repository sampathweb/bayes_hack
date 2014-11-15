from webassets import Environment, Bundle

my_env = Environment('static/', 'static/')

common_css = Bundle(
    'css/vendor/bootstrap.min.css',
    'css/vendor/sb-admin-2.css',
    'css/vendor/font-awesome.min.css',
    output='public/css/common.css'
)

common_js = Bundle(
    'js/vendor/jquery-2.1.1.js',
    'js/vendor/d3.v3.js',
    'js/vendor/mpld3.v0.2.js',
    output='public/js/common.js'
)
my_env.register('css_all', common_css)
my_env.register('js_all', common_js)
my_env['css_all'].urls()
my_env['js_all'].urls()
