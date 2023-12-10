import socketio
import eventlet
import random
import time
import json
import numpy as np
from racecar_gym_competition_env.racecar_gym.env import RaceEnv
from flask import Flask, jsonify, request, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
import io
import cv2


from multiprocessing import Process, Manager
import threading

import base64
from eventlet import wsgi
from werkzeug.wrappers import Request, Response

import matplotlib.pyplot as plt
from pathlib import Path

sio = socketio.Server()
app = socketio.WSGIApp(sio)

server_env = None

class ServerEnv:
    def __init__(self, sio):
        # socketio info
        self.sio = sio
        self.client_sid = None

        # env info
        self.env = RaceEnv(scenario="austria_competition",
                  render_mode='rgb_array_birds_eye',
                  reset_when_collision=True)
        
        
        # current env status
        self.obs, self.info = self.env.reset()
        self.reward = 0
        self.terminal = False
        self.trunc = None


        # trajcetory status 
        self.step = 0
        self.total_time = 0
        self.last_gettime = None
        

        # video status
        self.output_freq = 5
        # self.images = []

        # multiprocessing
        manager = Manager()
        self.images = manager.list()


    def get_obs_img(self):
        # img = self.env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
        img = self.env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
        # img = self.env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
        
        return img

    def get_img_views(self):
        progress = self.info['progress']
        lap = int(self.info['lap'])
        score = lap + progress - 1.

        # Get the images
        img1 = self.env.env.force_render(render_mode='rgb_array_higher_birds_eye', width=540, height=540,
                                    position=np.array([4.89, -9.30, -3.42]), fov=120)
        img2 = self.env.env.force_render(render_mode='rgb_array_birds_eye', width=270, height=270)
        img3 = self.env.env.force_render(render_mode='rgb_array_follow', width=128, height=128)
        img4 = (self.obs.transpose((1, 2, 0))).astype(np.uint8)

        # Combine the images
        img = np.zeros((540, 810, 3), dtype=np.uint8)
        img[0:540, 0:540, :] = img1
        img[:270, 540:810, :] = img2
        img[270 + 10:270 + 128 + 10, 540 + 7:540 + 128 + 7, :] = img3
        img[270 + 10:270 + 128 + 10, 540 + 128 + 14:540 + 128 + 128 + 14, :] = img4

        # Draw the text
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype('./racecar_gym/Arial.ttf', 25)
        # font_large = ImageFont.truetype('./racecar_gym/Arial.ttf', 35)
        draw.text((5, 5), "Full Map", fill=(255, 87, 34))
        draw.text((550, 10), "Bird's Eye", fill=(255, 87, 34))
        draw.text((550, 280), "Follow", fill=(255, 87, 34))
        draw.text((688, 280), "Obs", fill=(255, 87, 34))
        draw.text((550, 408), f"Lap {lap}", fill=(255, 255, 255))
        draw.text((688, 408), f"Prog {progress:.3f}", fill=(255, 255, 255))
        draw.text((550, 450), f"Score {score:.3f}", fill=(255, 255, 255))
        draw.text((550, 500), f"ID {self.client_sid}", fill=(255, 255, 255))
        img = np.asarray(img)

        self.images.append(img)
        
    
    def record_video(self, filename: str):
        height, width, layers = self.images[0].shape
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(filename, fourcc, 15, (width, height))
        for image in self.images:
            video.write(image)
        cv2.destroyAllWindows()
        video.release()

    def to_json(self, data):
        return json.dumps(data.tolist())
    
    def to_numpy(self, data):
        return np.array(json.loads(data))

    def get_obs(self, sid):
        try:
            self.sio.emit('push_obs', self.to_json(self.obs), to=sid)
            self.last_gettime = time.time()
        
        except Exception as e:
            print("get_obs", e)
            a = input()
        
    
    def set_action(self, action):
        action = self.to_numpy(action)
        print(action)
        try:
            self.total_time += time.time() - self.last_gettime
            # print(action)
            
            self.obs, _, terminal, self.trunc, info = self.env.step(action)
            # print(len(self.shared_list))
            
            progress = info['progress']
            lap = int(info['lap'])
            score = lap + progress - 1.
            env_time = info['time']

            # Print information
            # print_info = f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step: {self.step} Lap: {info["lap"]}, ' \
            #             f'Progress: {info["progress"]:.3f}, ' \
            #             f'EnvTime: {info["time"]:.3f} ' \
            #             f'AccTime: {self.total_time:.3f} '
            # if info.get('n_collision') is not None:
            #     print_info += f'Collision: {info["n_collision"]} '
            # if info.get('collision_penalties') is not None:
            #     print_info += 'CollisionPenalties: '
            #     for penalty in info['collision_penalties']:
            #         print_info += f'{penalty:.3f} '


            if self.step % self.output_freq == 0:
                p = Process(target=self.get_img_views)
                p.start()
            #     img = self.get_img_views()
            #     self.images.append(img)

            self.step += 1

            if terminal:
                if round(self.total_time) > MAX_ACCU_TIME:
                    print(f'[Time Limit Error] Accu time "{self.total_time}" violate the limit {MAX_ACCU_TIME} (sec)!')
                from datetime import datetime
                cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                video_name = f'results/{self.client_sid}_{cur_time}_env{env_time:.3f}_acc{round(self.total_time)}s_score{score:.4f}.mp4'
                Path(video_name).parent.mkdir(parents=True, exist_ok=True)
                self.record_video(video_name)
                print(f'============ Terminal ============')
                print(f'Video saved to {video_name}!')
                print(f'===================================')

                a = input()
                return jsonify({'terminal': terminal})
            


        except Exception as e:
            print("set_action", e)
            a = input()


    
    



@sio.event
def connect(sid, environ):
    if server_env.client_sid is None:
        server_env.client_sid = sid
        print(f"Client connected: {sid}")
    else:
        print("Another client is currently being handled.")


@sio.event
def greeting(sid, data):
    print(f"Received greeting from {sid}: {data}")
    send_obs(sid)


def send_obs(sid):
    server_env.get_obs(sid)
    

@sio.event
def get_action(sid, data):
    server_env.set_action(data)
    send_obs(sid)

@sio.event
def disconnect(sid):
    if server_env.client_sid and sid == server_env.client_sid:
        print(f"Client disconnected: {sid}")
        server_env = ServerEnv()



def custom_wsgi_app(env, start_response):
    path = env['PATH_INFO']
    if path == '/realtime':
        response = Response(render_realtime_page(), mimetype='text/html')
        return response(env, start_response)
    else:
        return app(env, start_response)

def image_to_base64(img_data):
    # if img_data.dtype != np.uint8:
    img_data = img_data.astype(np.uint8)
    img = Image.fromarray(img_data)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def send_image():
    while True:
        img_data = server_env.get_obs_img()  # 获取图像数据
        img_base64 = image_to_base64(img_data)  # 转换为 Base64
        sio.emit('new_image', {'image': img_base64})  # 发送到客户端
        # print("send image")
        eventlet.sleep(0.2)  # 每秒更新一次，可以根据需要调整
    


def render_realtime_page():
    # HTML 页面内容
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Realtime Page</title>
        <script src="https://cdn.jsdelivr.net/npm/socket.io/client-dist/socket.io.js"></script>
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function () {
                var socket = io();

                socket.on('new_image', function(data) {
                    document.getElementById('realtime-img').src = 'data:image/jpeg;base64,' + data.image;
                });
            });
        </script>
    </head>
    <body>
        <h1>Realtime Data</h1>
        <img id="realtime-img" src="" alt="Realtime Image">
    </body>
    </html>
    '''
    return html_content

if __name__ == '__main__':
    MAX_ACCU_TIME = 900
    server_env = ServerEnv(sio)
    eventlet.spawn(send_image)
    # realtimep = Process(target=send_image)
    # realtimep.start()
    
    

    eventlet.wsgi.server(eventlet.listen(('', 5000)), custom_wsgi_app)



