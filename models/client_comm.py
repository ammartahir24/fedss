import socket
import os
import json
import pickle
import jsonpickle
import random
from subprocess import Popen

class ClientComm:
    '''
    Server side part of network communication layer between server and client, launches and communicates with Client instance
    See run_client.py file for client part of the communication
    '''
    def __init__(self, user, group, train_data, test_data, model, ip, port, seed, lr, dataset):
        self.ip = ip
        self.port = port
        self.id = user
        self.group = group
        self.round_group = -1
        self.model = model
        self.model_params = []
        self.model_size_bytes = 0
        self.bandwidth = random.randint(3, 12)
        self.train_timeratio = round( 2.898 / random.randint(3, 12), 3)
        # write all data
        if not os.path.exists("temp"):
            os.mkdir("temp")
        data = {}
        data["user"] = user
        data["group"] = group
        data["train_data"] = train_data
        data["test_data"] = test_data
        data["seed"] = seed
        data["lr"] = lr
        data["model"] = model
        data["dataset"] = dataset
        # data["model"] = model
        # self.model.save("temp/"+ip+str(port)+"model")
        fname_data = "temp/" + ip + str(port) + ".json"
        cmd_client = "python3 run_client.py " + ip + " " + str(port) + " " + str(self.bandwidth) + " " + str(self.train_timeratio)
        with open(fname_data, 'w') as outfile:
            json.dump(data, outfile)
        # To Do: spawn process in mahimahi shell
        client_process = Popen(cmd_client, shell=True)
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # wait until client is up and running
        while True:
            try:
                soc.connect((self.ip, self.port))
                break
            except:
                continue
        
    def train(self, num_epochs, batch_size, minibatch, round_num):
        # send self.model and params to client
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("trn".encode('utf-8'))
        data = {}
        data["model"] = self.model_params
        data["num_epochs"] = num_epochs
        data["batch_size"] = batch_size
        data["minibatch"] = minibatch
        data["round_num"] = round_num
        data_to_send = json.dumps(jsonpickle.encode(data)).encode('utf-8')
        soc.send(str(len(data_to_send)).encode('utf-8'))
        soc.recv(2)
        soc.send(data_to_send)
    
    def test(self, set_to_use, round_num):
        # send self.model and set_to_use to client
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("tst".encode('utf-8'))
        data = {}
        data["model"] = self.model_params
        data["set_to_use"] = set_to_use
        data["round_num"] = round_num
        data_to_send = jsonpickle.encode(data).encode('utf-8')
        soc.send(str(len(data_to_send)).encode('utf-8'))
        soc.recv(2)
        soc.send(data_to_send)

    def stop(self):
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("stp".encode('utf-8'))
    
    def model_set_params(self, model_params):
        # print(model_params)
        self.model_params = model_params
    
    @property
    def num_samples(self):
        # get num_samples from client and return
        # print("num_samples requested on %d" %(self.port))
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("spl".encode("utf-8"))
        return int(soc.recv(1024).decode('utf-8'))

    @property
    def num_test_samples(self):
        # get num_samples from client and return
        # print("num_test_samples requested on %d" %(self.port))
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("ntt".encode("utf-8"))
        return int(soc.recv(1024).decode('utf-8'))

    @property
    def num_train_samples(self):
        # get num_samples from client and return
        # print("num_train_samples requested on %d" %(self.port))
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("ntr".encode("utf-8"))
        return int(soc.recv(1024).decode('utf-8'))

    @property
    def client_env(self):
        # The time(in ms) it takes to train one batch in one epoch
        data = {}
        data["train_timeratio"] = self.train_timeratio
        data["bandwidth"] = self.bandwidth
        # soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # soc.connect((self.ip, self.port))
        # soc.send("env".encode("utf-8"))
        # data = jsonpickle.decode(soc.recv(1024).decode("utf-8"))
        return data