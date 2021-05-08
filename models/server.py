import numpy as np
import socket
import threading
import queue
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
import jsonpickle
import json
import time
import random
from kneed import KneeLocator

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

    def evenify(self, client_groups, g_size):
        g_clients = {}
        for c,g in client_groups.items():
            if g in g_clients:
                g_clients[g].append(c)
            else:
                g_clients[g] = [c]
        # print(g_clients)
        for g in sorted(g_clients.keys()):
            # print([(g, len(g_clients[g])) for g in g_clients.keys()])
            if len(g_clients[g]) < g_size:
                print(g_clients[g])
                max_val = max(g_clients[g], key=lambda x:x.roundtime).roundtime
                while len(g_clients[g]) < g_size:
                    got_from_prev = False
                    for g_prev in range(g)[::-1]:
                        if len(g_clients[g_prev]) > g_size:
                            g_prev_max = max(g_clients[g_prev], key=lambda x:x.roundtime)
                            g_clients[g_prev].remove(g_prev_max)
                            g_clients[g].append(g_prev_max)
                            got_from_prev = True
                            break
                    if not got_from_prev:
                        got_from_next = False
                        for g_next in range(g,len(g_clients.keys())):
                            if len(g_clients[g_next]) > g_size:
                                g_next_min = min(g_clients[g_next], key=lambda x:x.roundtime)
                                if max_val < 1.5*g_next_min.roundtime:
                                    g_clients[g_next].remove(g_next_min)
                                    g_clients[g].append(g_next_min)
                                    got_from_next = True
                                    break
                        if not got_from_next:
                            # print(g, len(g_clients[g]))
                            next_gs = len(g_clients.keys()) - g - 1
                            elems_per_g = int(len(g_clients[g]) / next_gs)
                            c_elems = sorted(g_clients[g], key=lambda x:x.roundtime)
                            for e in range(next_gs):
                                g_clients[g+1+e] += c_elems[e*elems_per_g:e*(elems_per_g+1)]
                                g_clients[len(g_clients.keys())-1] += c_elems[(next_gs)*elems_per_g:-1]
                                g_clients[g] = []
                                break
        # print(g_clients)
        # print([(g, len(g_clients[g])) for g in g_clients.keys()])
        # print([(g, sorted(g_clients[g])) for g in g_clients.keys()])
        g_c_ret = {}
        for g,cs in g_clients.items():
            for c in cs:
                g_c_ret[c] = g
        return g_c_ret, g_clients

    def fedss_clustering(self, clients, num_groups):
        percentiles = [(i+1)/(num_groups+1) for i in range(num_groups)] # automate a way to find these
        model_size = len(json.dumps(jsonpickle.encode(self.model)).encode('utf-8'))
        c_to_times = {}
        c_to_groups = {}
        for c in clients:
            c_envs = c.client_env
            c_num_train_samples = c.num_train_samples
            download_time = model_size / (1000000*c_envs["bandwidth"]/8)
            upload_time = model_size / (1000000*(c_envs["bandwidth"]/2)/8)
            compute_time = (self.client_model.flops*c_num_train_samples)/(c_envs["flops"]*1000000)
            time_to_train = download_time + compute_time + upload_time
            c.roundtime = time_to_train
            # time_to_train = model_size / c_envs["bandwidth"] / 1000 + c_num_train_samples * c_envs["train_timeratio"]
            c_to_times[c] = time_to_train
            self.write_log("user: %s train_time: %d network: %d compute: %d ms model_size: %d\n" % \
                    (str(c.id), time_to_train, download_time+upload_time, compute_time, model_size) )
        times = sorted(list(c_to_times.values()))
        # print(times)
        time_percentiles = [times[int(p*len(times))] for p in percentiles]
        # print(time_percentiles)
        for c,t in c_to_times.items():
            c_to_groups[c] = self.closest_to(t, time_percentiles)
        c_to_groups, c_to_groups2 = self.evenify(c_to_groups, int(len(c_to_times.keys())/len(percentiles)))
        # groups = sorted(list(c_to_groups.values()))
        print([(c.roundtime, c_to_groups[c]) for c in c_to_groups.keys()])
        for c in clients:
            c.round_group = c_to_groups[c]
        # group_counts = {}
        # for i in range(max(groups)+1):
        #     group_counts[i] = groups.count(i)
        # min_percentage = min(list(group_counts.values())) / len(groups)
        # round_type_pattern = []
        # for k,v in group_counts.items():
        #     v_percentage = v / len(groups)
        #     v_ratio = int(0.5+ (v_percentage / min_percentage))
        #     round_type_pattern += [k]*v_ratio
        # print(round_type_pattern)
        # random.shuffle(round_type_pattern)
        # self.round_type_pattern = round_type_pattern
        return clients, c_to_groups, c_to_groups2

    def simulate(self, clusters, num_rounds, clients_per_round):
        pattern = [c for c in clusters.keys() if len(clusters[c])!=0]
        i = 0
        times = []
        for n in range(num_rounds):
            t = max(np.random.choice(clusters[pattern[i]], min(clients_per_round, len(clusters[pattern[i]])), replace=False), key=lambda x:x.roundtime).roundtime
            times.append(t)
            i = (i + 1) % len(pattern)
        k_anonymity = np.mean([1/len(clusters[p]) for p in pattern])
        avg_time = np.mean(times)
        return avg_time, k_anonymity

    def optimize_clusters(self, clients, max_clusters, clients_per_round):
        times = []
        k_anon = []
        for c in range(max_clusters)[::-1]:
            _, _, clusters = self.fedss_clustering(clients, c+1)
            avg_time, priv = self.simulate(clusters, 1000, clients_per_round)
            times.append(avg_time)
            k_anon.append(priv)
        kn = KneeLocator(times, k_anon, curve='convex', direction='decreasing')
        optimal = 3
        if kn.knee != None:
            optimal = max_clusters-times.index(kn.knee)
        print("Number of optimal clusters: %d" %(optimal))
        clients, _, clusters = self.fedss_clustering(clients, optimal)
        self.round_type_pattern = [c for c in clusters.keys() if len(clusters[c])!=0]
        random.shuffle(self.round_type_pattern)
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
        print("Pattern: %d" %(self.round_type_pattern[self.current_round_type]))
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
                   LOCAL_COMPUTATIONS_KEY: 0,
                   'time': 0} for c in clients}

        # TO DO: Add two loops: first distribute model to selected_clients, second waits for k responses
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
        sys_metrics = self.collect_updates(round_num, "train", k, len(clients), sys_metrics=sys_metrics)
        train_time = [ v['time'] for v in sys_metrics.values() ]
        train_time.sort()
        self.write_log("Round: %d time: %d ms\n" %(round_num, train_time[k-1]))
        return sys_metrics

    def collect_updates(self, round_num, type_, num_k, num_clients, sys_metrics={}):
        i = 0
        all_updates = []
        while i < num_clients:
            data = self.message_queue.get()
            if data['round_num'] < round_num:
                continue
            elif data['type'] == type_ and type_ == "test" and data['round_num'] == round_num:
                c_metrics, c_id = data['c_metrics'], data['id']
                sys_metrics[c_id] = c_metrics
                i += 1
            elif data['type'] == type_ and type_ == "train" and data['round_num'] == round_num:
                comp, num_samples, update, c_id, c_model_size, train_time = data['comp'], data['num_samples'], data['update'], data['id'], data['model_size'], data['time']
                sys_metrics[c_id][BYTES_READ_KEY] += c_model_size
                sys_metrics[c_id][BYTES_WRITTEN_KEY] += c_model_size
                sys_metrics[c_id][LOCAL_COMPUTATIONS_KEY] = comp
                sys_metrics[c_id]['time'] = train_time

                #self.updates.append((num_samples, update))
                all_updates.append((train_time, (num_samples, update), c_id))
                i += 1
            else:
                self.message_queue.put(data)
        if len(all_updates) != 0:
            all_updates.sort(key = lambda x : x[0])
            for j in range(num_k):
                #self.write_log("user: %s train_time: %d" % (all_updates[j][2], all_updates[j][0]))
                self.updates.append( all_updates[j][1] )
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

        metrics = self.collect_updates(num_round, "test", len(clients_to_test), len(clients_to_test), sys_metrics=metrics)
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
