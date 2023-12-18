import socketio
import eventlet
import random
import time

sio = socketio.Server()

@sio.event
def connect(sid, environ):
    print('connect ', sid)
    send_random_number(sid)

@sio.event
def hello(sid, data):
    print('hello from: ', data)
    

@sio.event
def number_received(sid, data):
    print('Number received from client: ', data)
    time.sleep(0.5)
    send_random_number(sid)

def send_random_number(sid):
    random_number = random.randint(1, 100)
    print('Sending number to client: ', random_number)
    sio.emit('number', random_number, to=sid)

app = socketio.WSGIApp(sio)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
