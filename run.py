from app import create_app


if __name__ == '__main__':
    env = 'hack'
    app = create_app('app.settings.%sConfig' % env.capitalize(), env)
    app.run(host='0.0.0.0', port=5000, debug=True)
