import sys
import json
from client import Client
import socket

ip, port = sys.argv[1], int(sys.argv[2])

data = {}
with open("temp/"+ip+str(port)+".json", 'r') as json_file:
    data = json.load(json_file)

user = data["user"]
group = data["group"]
train_data = data["train_data"]
test_data = data["test_data"]
model = data["model"]

client = Client(user, group, train_data, test_data, model)

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.bind((ip, port))
soc.listen()

while True:
    conn, addr = soc.accept()
    action = conn.recv(3)
    if action == "trn":
        # receive model, and params
        data = b''
        while True:
            d = conn.recv(1024)
            if not d:
                break
            data += d
        data = json.loads(data.decode('utf-8'))
        model = data['model']
        num_epochs = data['num_epochs']
        batch_size = data['batch_size']
        minibatch = data['minibatch']
        client.model.set_params(model)
        comp, num_samples, update = client.train(num_epochs, batch_size, minibatch)
        # send updates to server
    elif action == "tst":
        # receive model and param
        data = b''
        while True:
            d = conn.recv(1024)
            if not d:
                break
            data += d
        data = json.loads(data.decode('utf-8'))
        model = data['model']
        set_to_use = data['set_to_use']
        client.model.set_params(model)
        c_metrics = client.test(set_to_use)
        # send c_metrics to server
    elif action == "spl":
        conn.send(str(client.num_samples).encode('utf-8'))
    elif action == "stp":
        break