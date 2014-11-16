from app import create_app


def run_app():
    env = 'hack'
    app = create_app('app.settings.%sConfig' % env.capitalize(), env)
    return app

<<<<<<< HEAD
app = run_app()
=======
app = run_app()
>>>>>>> 9756699a294b68a8764d1751d0678fbfe5fa70be
