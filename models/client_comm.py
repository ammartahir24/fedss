import socket
import os
import json
import pickle
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
        self.model = model
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
        fname_log = "temp/" + ip + str(port) + ".log"
        cmd_client = "python3 run_client.py " + ip + " " + str(port) + " > " + fname_log
        with open(fname_data, 'w') as outfile:
            json.dump(data, outfile)
        # To Do: spawn process in mahimahi shell
        client_process = Popen(cmd_client, shell=True)
        
    def train(self, num_epochs, batch_size, minibatch, round_num):
        # send self.model and params to client
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("trn".encode('utf-8'))
        data = {}
        data["model"] = self.model
        data["num_epochs"] = num_epochs
        data["batch_size"] = batch_size
        data["minibatch"] = minibatch
        data["round_num"] = round_num
        soc.send(json.dumps(jsonpickle.encode(data)).encode('utf-8'))
    
    def test(self, set_to_use, round_num):
        # send self.model and set_to_use to client
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("tst".encode('utf-8'))
        data = {}
        data["model"] = self.model
        data["set_to_use"] = set_to_use
        data["round_num"] = round_num
        soc.send(json.dumps(jsonpickle.encode(data)).encode('utf-8'))

    def stop(self):
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("stp".encode('utf-8'))
    
    @property
    def num_samples(self):
        # get num_samples from client and return
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((self.ip, self.port))
        soc.send("spl".encode("utf-8"))
        return int(soc.recv(1023).decode('utf-8'))
