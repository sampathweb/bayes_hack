from app import create_app


def run_app():
    env = 'hack'
    app = create_app('app.settings.%sConfig' % env.capitalize(), env)
    return app

app = run_app()
