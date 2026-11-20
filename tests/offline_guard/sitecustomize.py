"""Explicit test-only transport denial; used with a cleared environment.

Never put this directory on PYTHONPATH for a real reader command.
"""
import os
import socket

def denied(*args, **kwargs):
    raise AssertionError('Offline test: network/resource transport forbidden')

socket.socket.connect = denied
socket.socket.connect_ex = denied
socket.socket.sendto = denied
socket.getaddrinfo = denied
socket.create_connection = denied
if os.environ.get('MODAL_CONFIG_PATH') != '/dev/null':
    raise RuntimeError('Offline tests require empty Modal credentials')
