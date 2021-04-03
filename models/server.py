import numpy as np
import socket
import threading
import queue
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
import jsonpickle
import json
import time
import random

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []
        self.ip = "127.0.0.1"
        self.port = 9999
        self.k = -1
        threading.Thread(target = self.listener, daemon=True).start()
        self.message_queue = queue.Queue()
        self.round_type_pattern = [0,0,1]
        self.current_round_type = 0
        log = open("temp/log.txt", 'w')
        log.write("%d\n" %(time.time()))
        log.close()

    def write_log(self, line):
        log = open("temp/log.txt", 'a')
        log.write(line)
        log.close()

    def listener(self):
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind((self.ip, self.port))
        soc.listen(20)
        while True:
            conn, addr = soc.accept()
            threading.Thread(target = self.handleConn, daemon=True, args=(conn,addr)).start()
    
    def msg_recv(self, conn, msg_len):
        recv_len = 0
        data = b''
        while recv_len < msg_len:
            d = conn.recv(msg_len)
            data += d
            recv_len += len(d)
        return data

    def handleConn(self, conn, addr):
        msg_len = int(conn.recv(1024).decode('utf-8'))
        conn.send("ok".encode('utf-8'))
        data = self.msg_recv(conn, msg_len)
        data = jsonpickle.decode(data.decode('utf-8'))
        self.message_queue.put(data)

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """ 

        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def closest_to(self, x, xs):
        x_xs = [abs(x_-x) for x_ in xs]
        return x_xs.index(min(x_xs))

    def create_round_pattern(self, clients):
        percentiles = [0.25, 0.5, 0.75] # automate a way to find these
        model_size = len(json.dumps(jsonpickle.encode(self.model)).encode('utf-8'))
        c_to_times = {}
        c_to_groups = {}
        for c in clients:
            c_envs = c.client_env
            c_num_train_samples = c.num_train_samples
            time_to_train = model_size / c_envs["bandwidth"] / 1000 + c_num_train_samples * c_envs["train_timeratio"]
            c_to_times[c.id] = time_to_train
        times = sorted(list(c_to_times.values()))
        print(times)
        time_percentiles = [times[int(p*len(times))] for p in percentiles]
        print(time_percentiles)
        for c,t in c_to_times.items():
            c_to_groups[c] = self.closest_to(t, time_percentiles)
        groups = sorted(list(c_to_groups.values()))
        print(groups)
        for c in clients:
            c.round_group = c_to_groups[c.id]
        group_counts = {}
        for i in range(max(groups)+1):
            group_counts[i] = groups.count(i)
        min_percentage = min(list(group_counts.values())) / len(groups)
        round_type_pattern = []
        for k,v in group_counts.items():
            v_percentage = v / len(groups)
            v_ratio = int(0.5+ (v_percentage / min_percentage))
            round_type_pattern += [k]*v_ratio
        print(round_type_pattern)
        random.shuffle(round_type_pattern)
        self.round_type_pattern = round_type_pattern
        return clients

    def smart_select_clients(self, my_round, clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        # samples_dict = {}
        # envs_dict = {}
        # time_dict = {}
        # model_size = len(json.dumps(jsonpickle.encode(self.model)).encode('utf-8'))
        # for c in possible_clients:
        #     samples_dict[c] = (c.num_train_samples, c.num_test_samples)
        #     envs_dict[c] = c.client_env
        #     time_dict[c] = model_size / envs_dict[c]["bandwidth"] / 1000 + samples_dict[c][0] * envs_dict[c]["train_timeratio"]
        #     print("user: %s num_train: %d num_test: %d model_size: %d time: %d ms" % \
        #         (c.id, samples_dict[c][0], samples_dict[c][1], model_size, time_dict[c]))
        #     print(envs_dict[c])
        possible_clients = [c for c in clients if c.round_group == self.round_type_pattern[self.current_round_type]]
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
        if len(self.selected_clients) < num_clients:
            possible_clients_backup = [c for c in clients if c.round_group < self.round_type_pattern[self.current_round_type]]
            self.selected_clients += np.random.choice(possible_clients_backup, num_clients - len(self.selected_clients), replace=False)
        self.current_round_type = (self.current_round_type + 1) % len(self.round_type_pattern)
        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None, round_num=0, k=-1):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        
        # TO DO: Add two loops: first distribute model to selected_clients, second waits for k responses
        ts_start = time.time()
        for c in clients:
            c.model_set_params(self.model)
            c.train(num_epochs, batch_size, minibatch, round_num)
            # comp, num_samples, update = c.train(num_epochs, batch_size, minibatch, round_num)

            # sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            # sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            # sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            # self.updates.append((num_samples, update))
        if k==-1:
            k = len(clients)
        sys_metrics = self.collect_updates(round_num, "train", min(k, len(clients)), sys_metrics=sys_metrics)
        ts_end = time.time()
        round_time = ts_end - ts_start
        self.write_log("Round time for round no. %d: %f\n" %(round_num, round_time))
        return sys_metrics

    def collect_updates(self, round_num, type_, num_clients, sys_metrics={}):
        num_k = 0
        while num_k < num_clients:
            data = self.message_queue.get()
            if data['round_num'] < round_num:
                continue
            elif data['type'] == type_ and type_ == "test" and data['round_num'] == round_num:
                c_metrics, c_id = data['c_metrics'], data['id']
                sys_metrics[c_id] = c_metrics
                num_k += 1
            elif data['type'] == type_ and type_ == "train" and data['round_num'] == round_num:
                comp, num_samples, update, c_id, c_model_size = data['comp'], data['num_samples'], data['update'], data['id'], data['model_size']
                sys_metrics[c_id][BYTES_READ_KEY] += c_model_size
                sys_metrics[c_id][BYTES_WRITTEN_KEY] += c_model_size
                sys_metrics[c_id][LOCAL_COMPUTATIONS_KEY] = comp

                self.updates.append((num_samples, update))
                num_k += 1
            else:
                self.message_queue.put(data)
        return sys_metrics

    def update_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def test_model(self, clients_to_test, num_round, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model_set_params(self.model)
            client.test(set_to_use, num_round)
            # c_metrics = client.test(set_to_use)
            # metrics[client.id] = c_metrics
        
        metrics = self.collect_updates(num_round, "test", len(clients_to_test), sys_metrics=metrics)
        for k,v in metrics.items():
            self.write_log("%d round@ %s for %s data: accuracy = %f, loss = %f\n" %(num_round, k, set_to_use, v['accuracy'], v['loss']))
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()