#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)

sys.path.insert(0, "/usr/local/miniconda/bin")
<<<<<<< HEAD
sys.path.insert(0,"/data/bayes/bayes_hack/bayes-hack-app/")
=======
sys.path.insert(0, "/data/bayes/bayes_hack/bayes-hack-app/")
>>>>>>> 9756699a294b68a8764d1751d0678fbfe5fa70be

from app import create_app

env = 'hack'
application = create_app('app.settings.%sConfig' % env.capitalize(), env)

application.secret_key = 'fefwoafkawj03jt023jaaj@!Rrok2po'


