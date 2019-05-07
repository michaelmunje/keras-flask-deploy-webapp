FROM python:2.7.16-slim-stretch

RUN pip install Werkzeug Flask numpy Keras gevent pillow h5py tensorflow

EXPOSE 5000

