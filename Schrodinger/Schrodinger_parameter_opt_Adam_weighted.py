import sys
import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub, learning_rate, weighted):

        X0 = np.concatenate((x0, 0 * x0), 1) 
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1) 
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1) 

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]  # t0=0

        self.x_lb = X_lb[:, 0:1]  # x_lb=lb[0]
        self.t_lb = X_lb[:, 1:2]  # t_lb=tb

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0

        self.learning_rate = tf.constant(learning_rate)
        self.weighted = weighted

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        self.saver = tf.train.Saver(max_to_keep=30000)

        # tf placeholders and graph
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v0_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs
        self.u0_pred, self.v0_pred, _, _ = self.net_uv(self.x0_tf, self.t0_tf)
        self.u_lb_pred, self.v_lb_pred, self.u_x_lb_pred, self.v_x_lb_pred = self.net_uv(self.x_lb_tf,
                                                                                         self.t_lb_tf)
        self.u_ub_pred, self.v_ub_pred, self.u_x_ub_pred, self.v_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_u_pred, self.f_v_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

        # Loss
        self.loss_0 = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                     tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred))
        self.loss_b = tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                     tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                     tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                     tf.reduce_mean(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_u_pred)) + \
                     tf.reduce_mean(tf.square(self.f_v_pred))
        self.mean_mape_0 = tf.reduce_mean(tf.abs(self.u0_tf - self.u0_pred) / tf.abs(self.u0_tf))
        self.loss = (self.loss_0 + self.loss_b) / (1 + self.weighted) + self.loss_f * (
                    self.weighted / 1 + self.weighted)

        self.var_0 = tf.math.reduce_variance(tf.square(self.u0_tf - self.u0_pred))
        self.var_b = tf.math.reduce_variance(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                     tf.math.reduce_variance(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                     tf.math.reduce_variance(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                     tf.math.reduce_variance(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
        self.var_f = tf.math.reduce_variance(tf.square(self.f_u_pred)) + tf.math.reduce_variance(tf.square(self.f_v_pred))
        self.var_mape_0 = tf.math.reduce_variance(tf.abs(self.u0_tf - self.u0_pred) / tf.abs(self.u0_tf))

        self.worst_0 = tf.math.reduce_max(tf.square(self.u0_tf  - self.u0_pred))
        self.worst_b = tf.math.reduce_max(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                     tf.math.reduce_max(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
                     tf.math.reduce_max(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                     tf.math.reduce_max(tf.square(self.v_x_lb_pred - self.v_x_ub_pred))
        self.worst_f =  tf.math.reduce_max(tf.square(self.f_u_pred)) + tf.math.reduce_max(tf.square(self.f_v_pred))
        self.worst_mape_0 = tf.reduce_max(tf.abs(self.u0_tf - self.u0_pred) / tf.abs(self.u0_tf))

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            self.W = W
            self.b = b
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y

    def net_uv(self, x, t):
        X = tf.concat([x, t], 1) 

        uv = self.neural_net(X, self.weights, self.biases) 
        u = uv[:, 0:1] 
        v = uv[:, 1:2]  

        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        u, v, u_x, v_x = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]

        f_u = u_t + 0.5 * v_xx + (
                u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return f_u, f_v

    def callback(self, loss):
        print(loss)


    def train(self, nIter, n_outlook, diff_loss, X_val_star, u_val_star, v_val_star, h_val_star, num_layers, num_neurons, learning_rate, weighted):

        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0, self.v0_tf: self.v0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        n_epoch = []

        val_err = []

        loss_of_all = []
        loss_of_0 = []
        loss_of_b = []
        loss_of_f = []

        mean_of_epoch_0 = []
        mean_of_epoch_b = []
        mean_of_epoch_f = []
        mean_of_mape_0 = []

        var_of_epoch_0 = []
        var_of_epoch_b = []
        var_of_epoch_f = []
        var_of_mape_0 = []


        worst_of_epoch_0 = []
        worst_of_epoch_b = []
        worst_of_epoch_f = []
        worst_of_mape_0 = []

        best_val = np.inf
        start_time = time.time()
        for it in range(nIter):
            _, new_loss, loss_0, loss_b, loss_f = self.sess.run([self.train_op_Adam, self.loss, self.loss_0, self.loss_b, self.loss_f], tf_dict)

            loss_of_all.insert(it, self.sess.run(self.loss, tf_dict))
            loss_of_0.insert(it, loss_0)
            loss_of_b.insert(it, loss_b)
            loss_of_f.insert(it, loss_f)

            mean_of_epoch_0.insert(it, self.sess.run(self.loss_0, tf_dict))
            mean_of_epoch_b.insert(it, self.sess.run(self.loss_b, tf_dict))
            mean_of_epoch_f.insert(it, self.sess.run(self.loss_f, tf_dict))
            mean_of_mape_0.insert(it, self.sess.run(self.mean_mape_0, tf_dict))

            var_of_epoch_0.insert(it, self.sess.run(self.var_0, tf_dict))
            var_of_epoch_b.insert(it, self.sess.run(self.var_b, tf_dict))
            var_of_epoch_f.insert(it, self.sess.run(self.var_f, tf_dict))
            var_of_mape_0.insert(it, self.sess.run(self.var_mape_0, tf_dict))

            worst_of_epoch_0.insert(it, self.sess.run(self.worst_0, tf_dict))
            worst_of_epoch_b.insert(it, self.sess.run(self.worst_b, tf_dict))
            worst_of_epoch_f.insert(it, self.sess.run(self.worst_f, tf_dict))
            worst_of_mape_0.insert(it, self.sess.run(self.worst_mape_0, tf_dict))

            n_epoch.insert(it, it)

            if it >= n_outlook:
                if abs(max(loss_of_all[it - n_outlook: it]) - min(loss_of_all[it - n_outlook: it])) < diff_loss:
                    break
#
            if it % n_outlook == 0:
                self.saver.save(self.sess,
                                './tf_model/Schrodinger-Adam-weighted-OriginalNN-%d-%d-%.4f-%d-%d.ckpt' % (
                                    num_layers, num_neurons, learning_rate, weighted, it))
                u_tf_pred, v_tf_pred = self.predict_val(X_val_star)
                np.linalg.norm(u_val_star - u_tf_pred, 2) / np.linalg.norm(u_val_star, 2)
                np.linalg.norm(v_val_star - v_tf_pred, 2) / np.linalg.norm(v_val_star, 2)
                h_tf_pred = np.sqrt(u_tf_pred ** 2 + v_tf_pred ** 2)
                val_error = np.linalg.norm(h_val_star - h_tf_pred, 2) / np.linalg.norm(h_val_star, 2)
                val_err.insert(it, val_error)

                if best_val > val_error:
                    best_val = val_error
                    i = it
                    self.saver.save(self.sess,'./tf_model/Schrodinger-Adam-weighted-OriginalNN-%d-%d-%.4f-%d-%d.ckpt' % (
                                                            num_layers, num_neurons, learning_rate, weighted))
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, new_loss, elapsed))
                start_time = time.time()

        return loss_of_all, loss_of_0, loss_of_b, loss_of_f, mean_of_epoch_0, mean_of_epoch_b, mean_of_epoch_f, mean_of_mape_0, var_of_epoch_0, var_of_epoch_b, var_of_epoch_f, var_of_mape_0,\
               worst_of_epoch_0, worst_of_epoch_b, worst_of_epoch_f, worst_of_mape_0, n_epoch, best_val, val_err, i


    def predict_val(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star


    def predict(self, X_star, it):

        self.saver.restore(self.sess, './tf_model/Schrodinger-Adam-weighted-OriginalNN-%d-%d-%.4f-%d-%d.ckpt'
                                    % (num_layers, num_neurons, learning_rate, weighted, it))
        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)
        v_star = self.sess.run(self.v0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)
        f_v_star = self.sess.run(self.f_v_pred, tf_dict)

        return u_star, v_star, f_u_star, f_v_star



def main_loop(num_layers, num_neurons, learning_rate, weighted):
    
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 4])

    N0 = 50
    N_b = 50
    N_f = 20000

    num_layers = num_layers
    num_neurons = num_neurons
    learning_rate = learning_rate
    weighted = weighted

    layers = np.concatenate([[2], num_neurons * np.ones(num_layers), [2]]).astype(int).tolist()

    data = scipy.io.loadmat('../Data/NLS.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    len1 = len(t[t <= np.pi / 4])
    len2 = len(t[t <= np.pi * (2/5)])

    t_train = t[0:len1, :]  
    t_val = t[len1:len2, :] 
    t_test = t[len2:, :]  

    Exact_u_train = Exact_u[:, 0:len1]
    Exact_u_val = Exact_u[:, len1:len2]
    Exact_u_test = Exact_u[:, len2:]

    Exact_v_train = Exact_v[:, 0:len1]
    Exact_v_val = Exact_v[:, len1:len2]
    Exact_v_test = Exact_v[:, len2:]

    Exact_h_train = np.sqrt(Exact_u_train ** 2 + Exact_v_train ** 2)
    Exact_h_val = np.sqrt(Exact_u_val ** 2 + Exact_v_val ** 2)
    Exact_h_test = np.sqrt(Exact_u_test ** 2 + Exact_v_test ** 2)

    X, T = np.meshgrid(x, t)
    X_train, T_train = np.meshgrid(x, t_train)
    X_val, T_val = np.meshgrid(x, t_val)
    X_test, T_test = np.meshgrid(x, t_test)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    X_train_star = np.hstack((X_train.flatten()[:, None], T_train.flatten()[:, None]))
    X_val_star = np.hstack((X_val.flatten()[:, None], T_val.flatten()[:, None]))
    X_test_star = np.hstack((X_test.flatten()[:, None], T_test.flatten()[:, None]))

    u_star = Exact_u.T.flatten()[:, None]
    u_train_star = Exact_u_train.T.flatten()[:, None]
    u_val_star = Exact_u_val.T.flatten()[:, None]
    u_test_star = Exact_u_test.T.flatten()[:, None]

    v_star = Exact_v.T.flatten()[:, None]
    v_train_star = Exact_v_train.T.flatten()[:, None]
    v_val_star = Exact_v_val.T.flatten()[:, None]
    v_test_star = Exact_v_test.T.flatten()[:, None]

    h_star = Exact_h.T.flatten()[:, None]
    h_train_star = Exact_h_train.T.flatten()[:, None]
    h_val_star = Exact_h_val.T.flatten()[:, None]
    h_test_star = Exact_h_test.T.flatten()[:, None]

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t_train.shape[0], N_b, replace=False)
    tb = t_train[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub, learning_rate, weighted)

    start_time = time.time()
    loss_of_all, loss_0, loss_b, loss_f, mean_epoch_of_0, mean_epoch_of_b, mean_epoch_of_f, mean_epoch_of_mape_0, var_epoch_of_0, var_epoch_of_b, var_epoch_of_f, var_epoch_of_mape_0, \
    worst_epoch_of_0, worst_epoch_of_b, worst_epoch_of_f, worst_epoch_of_mape_0, epoch, error_h_extra, validation_error, best_it = \
        model.train(30000, 50, 0.00001, X_val_star, u_val_star, v_val_star, h_val_star, num_layers, num_neurons, learning_rate, weighted)
    u_pred_inter, v_pred_inter, f_u_pred_inter, f_v_pred_inter = model.predict(X_train_star, best_it)
    u_pred_test, v_pred_test, f_u_pred_test, f_v_pred_test = model.predict(X_test_star, best_it)

    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    h_pred_inter = np.sqrt(u_pred_inter ** 2 + v_pred_inter ** 2)
    h_pred_test = np.sqrt(u_pred_test ** 2 + v_pred_test ** 2)

    error_h_inter = np.linalg.norm(h_train_star - h_pred_inter, 2) / np.linalg.norm(h_train_star, 2)
    print('Error h: %e' % (error_h_inter))

    print('Error h: %e' % (error_h_extra))

    error_h_test = np.linalg.norm(h_test_star - h_pred_test, 2) / np.linalg.norm(h_test_star, 2)
    print('Error h: %e' % (error_h_test))


    return error_h_inter, error_h_extra, error_h_test, best_it


if __name__ == "__main__":
    
    num_layers = int(sys.argv[1])
    num_neurons = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    weighted = int(sys.argv[4])
    
    # num_layers = 5
    # num_neurons = 150
    # learning_rate = 0.0005
    # weighted = 10

    result1 = main_loop(num_layers, num_neurons, learning_rate, weighted)
    print(result1)
