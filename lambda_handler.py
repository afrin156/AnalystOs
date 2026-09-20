"""
Lambda entry point - wraps the same Flask app used locally, so nothing
behaves differently once deployed. Used only by `sam deploy`.
"""

import awsgi
from app import app


def handler(event, context):
    return awsgi.response(app, event, context)
