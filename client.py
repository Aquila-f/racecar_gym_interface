import socketio
import numpy as np
import json

# 創建一個 Socket.IO 客戶端
sio = socketio.Client()


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
        
# Initialize the RL Agent
import gymnasium as gym

agent = RandomAgent(
    action_space=gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32))


def to_json(data):
    return json.dumps(data.tolist())

def to_numpy(data):
    return np.array(json.loads(data))

def send_act(data):
    sio.emit('get_action', to_json(data))

@sio.event
def push_obs(data):
    action_to_take = agent.act(np.array(json.loads(data)))  # Replace with actual action
    send_act(action_to_take)





# 當客戶端連接到服務器時觸發的事件
@sio.event
def connect():
    print("Connected to server.")
    sio.emit('greeting', 'Hello from client!')


@sio.event
def disconnect():
    print("Disconnected from server.")

# 連接到服務器
sio.connect('http://localhost:5000')
sio.wait()

# 發送消息給服務器
# try:
#     while True:
#         inp_num = input("Enter a message (type 'exit' to stop): ")
#         # if message == 'exit':
#         #     break
#         sio.emit('number', inp_num)
# finally:
#     sio.disconnect()
