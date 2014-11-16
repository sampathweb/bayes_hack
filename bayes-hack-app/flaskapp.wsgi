#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0, "/data/bayes/bayes_hack/bayes-hack-app/")

from app import create_app

env = 'hack'
application = create_app('app.settings.%sConfig' % env.capitalize(), env)

application.secret_key = 'fefwoafkawj03jt023jaaj@!Rrok2po'


