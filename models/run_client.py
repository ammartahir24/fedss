import sys
import json
from client import Client
import socket
import pickle
import os
import importlib
import random
import jsonpickle
import tensorflow as tf
import numpy as np
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
# from tensorflow import keras

'''
Runs Client instance and manages all its communication with the server (managed by ClientComm)
'''
ip, port = sys.argv[1], int(sys.argv[2])
server_ip, server_port = "127.0.0.1", 9999 # replace 127.0.0.1 with mahimahi assigned IP?
data = {}
with open("temp/"+ip+str(port)+".json", 'r') as json_file:
    data = json.load(json_file)

user = data["user"]
group = data["group"]
train_data = data["train_data"]
test_data = data["test_data"]
seed = data["seed"]
lr = data["lr"]
model = data["model"]
dataset = data["dataset"]

random.seed(1 + seed)
np.random.seed(12 + seed)
tf.set_random_seed(123 + seed)

model_path = '%s/%s.py' % (dataset, model)
if not os.path.exists(model_path):
    print('Please specify a valid dataset and a valid model.')
model_path = '%s.%s' % (dataset, model)

mod = importlib.import_module(model_path)
ClientModel = getattr(mod, 'ClientModel')

tf.logging.set_verbosity(tf.logging.WARN)

model_params = MODEL_PARAMS[model_path]
if lr != -1:
    model_params_list = list(model_params)
    model_params_list[0] = lr
    model_params = tuple(model_params_list)

tf.reset_default_graph()
client_model = ClientModel(seed, *model_params)

# model = keras.models.load_model("temp/"+ip+str(port)+"model")

client = Client(user, group, train_data, test_data, client_model)


def msg_recv(conn, msg_len):
    recv_len = 0
    data = b''
    while recv_len < msg_len:
        d = conn.recv(msg_len)
        data += d
        recv_len += len(d)
    return data

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.bind((ip, port))
soc.listen()
print('--- Client running at (%s, %d) ---' % (ip, port))

while True:
    conn, addr = soc.accept()
    action = conn.recv(3).decode('utf-8')
    print("%s message received from %d" %(action, addr[1]))
    if action == "trn":
        # receive model, and params
        msg_len = int(conn.recv(1024).decode('utf-8'))
        conn.send("ok".encode('utf-8'))
        data = msg_recv(conn, msg_len)
        data = jsonpickle.decode(json.loads(data.decode('utf-8')))
        model = data['model']
        num_epochs = data['num_epochs']
        batch_size = data['batch_size']
        minibatch = data['minibatch']
        round_num = data['round_num']
        print("%d: model received, starting training" %(port))
        client.model.set_params(model)
        comp, num_samples, update = client.train(num_epochs, batch_size, minibatch)
        # send updates to server
        print("%d: training complete, sending weights to server" %(port))
        svr_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        svr_soc.connect((server_ip, server_port))
        data = {}
        data['comp'] = comp
        data['num_samples'] = num_samples
        data['update'] = update
        data['round_num'] = round_num
        data['type'] = 'train'
        data['id'] = client.id
        data['model_size'] = client.model.size
        data_to_send = jsonpickle.encode(data).encode('utf-8')
        svr_soc.send(str(len(data_to_send)).encode('utf-8'))
        svr_soc.recv(2)
        svr_soc.send(data_to_send)
    elif action == "tst":
        # receive model and param
        msg_len = int(conn.recv(1024).decode('utf-8'))
        conn.send("ok".encode('utf-8'))
        print("%d: receiving %d bytes" %(port, msg_len))
        data = msg_recv(conn, msg_len)
        assert(len(data) == msg_len)
        data = jsonpickle.decode(data.decode('utf-8'))
        model = data['model']
        set_to_use = data['set_to_use']
        round_num = data['round_num']
        print("%d: model received, starting testing" %(port))
        client.model.set_params(model)
        c_metrics = client.test(set_to_use)
        print("%d: testing complete, sending metrics to server" %(port))
        # send c_metrics to server
        svr_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        svr_soc.connect((server_ip, server_port))
        data = {}
        data['c_metrics'] = c_metrics
        data['round_num'] = round_num
        data['id'] = client.id
        data['type'] = 'test'
        data_to_send = jsonpickle.encode(data).encode('utf-8')
        svr_soc.send(str(len(data_to_send)).encode('utf-8'))
        svr_soc.recv(2)
        svr_soc.send(data_to_send)
        print("%d: sent metrics to server" %(port))
    elif action == "spl":
        conn.send(str(client.num_samples).encode('utf-8'))
    elif action == "ntt":
        conn.send(str(client.num_test_samples).encode('utf-8'))
    elif action == "ntr":
        conn.send(str(client.num_train_samples).encode('utf-8'))
    elif action == "stp":
        break

# class ClientAsync:
#     def __init__():
#         pass

# if __name__ == "__main__":
#     print("--- Client process initiated")
#     ip, port = sys.argv[1], int(sys.argv[2])
#     server_ip, server_port = "127.0.0.1", 9999
#     soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     soc.bind((ip, port))
#     soc.listen()
#     print('--- Client running at (%s, %d) ---' % (ip, port))
#     while True:
#         conn, addr = soc.accept()
#         data_binary = b''
#         while True:
#             d = conn.recv(1024)
#             if not d:
#                 break
#             data_binary += d
#         print(data_binary)