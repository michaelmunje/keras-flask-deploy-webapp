#!/bin/bash
docker container stop $(docker ps -aq -f "ancestor=keras_flask_app")
