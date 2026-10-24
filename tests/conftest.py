"""The unit suite must not contact live datasets or model providers."""
import socket
import pytest

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in unit tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
