import socketio
import time

sio = socketio.Client()

@sio.event
def connect():
    print('Connection established')
    sio.emit('hello', 'client')

@sio.event
def number(data):
    print('Number received from server: ', data)
    new_number = data - 10
    print('Sending number to server: ', new_number)
    time.sleep(0.5)
    sio.emit('number_received', new_number)
    

@sio.event
def disconnect():
    print('Disconnected from server')

sio.connect('http://localhost:5000')

sio.wait()
