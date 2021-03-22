import socket
import os
import json
from subprocess import Popen


class ClientComm:
    def __init__(user, group, train_data, test_data, model, ip, port):
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
        data["model"] = model
        with open("temp/"+ip+str(port)+".json", 'w') as outfile:
            json.dump(data, outfile)
        #spawn client process
        # To Do: spawn process in mahimahi shell
        client_process = Popen("python3 run_client.py "+ip+" "+str(port), shell=True)
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((self.ip, self.port))

        def train(self, num_epochs, batch_size, minibatch):
            # send self.model and params to client
            self.soc.send("trn".encode('utf-8'))
            data = {}
            data["model"] = self.model
            data["num_epochs"] = num_epochs
            data["batch_size"] = batch_size
            data["minibatch"] = minibatch
            self.soc.send(json.dumps(data).encode('utf-8'))
        
        def test(self, set_to_use):
            # send self.model and set_to_use to client
            self.soc.send("tst".encode('utf-8'))
            data = {}
            data["model"] = self.model
            data["set_to_use"] = set_to_use
            self.soc.send(json.dumps(data).encode('utf-8'))

        def stop(self):
            self.soc.send("stp".encode('utf-8'))
        
        @property
        def num_samples(self):
            # get num_samples from client and return
            self.soc.send("spl".encode("utf-8"))
            return int(self.soc.recv(1023).decode('utf-8'))
