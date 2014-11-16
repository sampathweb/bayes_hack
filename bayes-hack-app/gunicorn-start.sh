#!/bin/bash

NAME="DonorsChooseApp"
SOCKFILE=/home/ramesh/sock
USER=ramesh
GROUP=staff
NUM_WORKERS=3

echo "Starting $NAME"

export PYTHONPATH=/usr/local/miniconda/bin

# Create the run directory if it doesn't exist
RUNDIR=$(dirname $SOCKFILE)
test -d $RUNDIR || mkdir -p $RUNDIR

# Start your unicorn
exec gunicorn run:app -b 0.0.0.0:8000 \
  --name $NAME \
  --workers $NUM_WORKERS \
  --user=$USER --group=$GROUP \
  --log-level=debug \

#   --bind=unix:$SOCKFILE
