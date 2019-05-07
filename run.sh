#!/bin/bash
docker run -ti --rm -v -d -p 5000:5000 -v $(pwd):/src:rw keras_flask_app bash -c "cd src; python app.py"
