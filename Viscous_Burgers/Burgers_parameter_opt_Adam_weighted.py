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
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, learning_rate, weighted):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu

        self.learning_rate = tf.constant(learning_rate)
        self.weighted = weighted

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        self.saver = tf.train.Saver(max_to_keep=30000)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred))
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred))
        self.loss = self.loss_u/(1 + self.weighted) + self.loss_f*(self.weighted/(1+self.weighted))

        self.mape_u = tf.reduce_mean(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))

        self.var_u = tf.math.reduce_variance(tf.square(self.u_tf - self.u_pred))
        self.var_f = tf.math.reduce_variance(tf.square(self.f_pred))
        self.var_mape_u = tf.math.reduce_variance(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))

        self.worst_u = tf.math.reduce_max(tf.square(self.u_tf - self.u_pred))
        self.worst_f = tf.math.reduce_max(tf.square(self.f_pred))
        self.worst_mape_u = tf.reduce_max(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))

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

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def callback(self, loss):
        print('Loss', loss)

    def train(self, nIter, n_outlook, diff_loss, X_val_star, u_val_star, num_layers, num_neurons, learning_rate, weighted):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        n_epoch = []

        val_err = []

        loss_of_all = []
        loss_of_u = []
        loss_of_f = []

        mean_of_epoch_u = []
        mean_of_epoch_f = []
        mean_of_mape_u = []

        var_of_epoch_u = []
        var_of_epoch_f = []
        var_of_mape_u = []

        worst_of_epoch_u = []
        worst_of_epoch_f = []
        worst_of_mape_u = []

        best_val = np.inf
        start_time = time.time()
        
        for it in range(nIter):

            _, new_loss, lossu, lossf = self.sess.run([self.train_op_Adam, self.loss, self.loss_u, self.loss_f], tf_dict)

            loss_of_all.insert(it, new_loss)
            loss_of_u.insert(it, lossu)
            loss_of_f.insert(it, lossf)

            mean_of_epoch_u.insert(it, self.sess.run(self.loss_u, tf_dict))
            mean_of_epoch_f.insert(it, self.sess.run(self.loss_f, tf_dict))
            mean_of_mape_u.insert(it, self.sess.run(self.mape_u, tf_dict))

            var_of_epoch_u.insert(it, self.sess.run(self.var_u, tf_dict))
            var_of_epoch_f.insert(it, self.sess.run(self.var_f, tf_dict))
            var_of_mape_u.insert(it, self.sess.run(self.var_mape_u, tf_dict))

            worst_of_epoch_u.insert(it, self.sess.run(self.worst_u, tf_dict))
            worst_of_epoch_f.insert(it, self.sess.run(self.worst_f, tf_dict))
            worst_of_mape_u.insert(it, self.sess.run(self.worst_mape_u, tf_dict))

            n_epoch.insert(it, it)

            if it >= n_outlook:
                if abs(max(loss_of_all[it - n_outlook: it]) - min(loss_of_all[it - n_outlook: it])) < diff_loss:
                    break

            if it % n_outlook == 0:
                self.saver.save(self.sess,
                                './tf_model/Viscous_burgers-Adam-weighted-OriginalNN-%d-%d-%.4f-%d-%d.ckpt' % (
                                    num_layers, num_neurons, learning_rate, weighted, it))
                u_tf_pred, f_tf_pred = self.predict_val(X_val_star)
                val_error = np.linalg.norm(u_val_star - u_tf_pred, 2) / np.linalg.norm(u_val_star, 2)
                val_err.insert(it, val_error)

                if best_val > val_error:
                    best_val = val_error
                    i = it
                    self.saver.save(self.sess,'./tf_model/Viscous_burgers-Adam-weighted-OriginalNN-%d-%d-%.4f-%d.ckpt' % (
                                                            num_layers, num_neurons, learning_rate, weighted))
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, new_loss, elapsed))
                start_time = time.time()

        return loss_of_all, loss_of_u, loss_of_f, mean_of_epoch_u, mean_of_epoch_f, mean_of_mape_u, var_of_epoch_u, var_of_epoch_f, \
               var_of_mape_u, worst_of_epoch_u, worst_of_epoch_f, worst_of_mape_u, n_epoch, best_val, val_err, i

    def predict_val(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star

    def predict(self, X_star, it):
        self.saver.restore(self.sess, './tf_model/Viscous_burgers-Adam-weighted-OriginalNN-%d-%d-%.4f-%d-%d.ckpt' % (
                    num_layers, num_neurons, learning_rate, weighted, it))
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star



def main_loop(num_layers, num_neurons, learning_rate, weighted):

    N_u = 100
    N_f  = 10000

    nu = 0.01 / np.pi

    num_layers = num_layers
    num_neurons = num_neurons
    learning_rate = learning_rate
    weighted = weighted

    layers = np.concatenate([[2], num_neurons * np.ones(num_layers), [1]]).astype(int).tolist()

    data = scipy.io.loadmat("../../Data/burgers_shock.mat")

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    len1 = len(t[t <= 0.5]) 
    len2 = len(t[t <= 0.8])
    t_train = t[0: len1, :]
    t_val = t[len1:len2, :] 
    t_test = t[len2:, :]

    Exact_train = Exact[: len1, :]
    Exact_val = Exact[len1:len2, :] 
    Exact_test = Exact[len2:, :]

    X, T = np.meshgrid(x, t)
    X_train, T_train = np.meshgrid(x, t_train)
    X_val, T_val = np.meshgrid(x, t_val)
    X_test, T_test = np.meshgrid(x, t_test)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    X_tr_star = np.hstack((X_train.flatten()[:, None], T_train.flatten()[:, None]))
    u_tr_star = Exact_train.flatten()[:, None]
    X_val_star = np.hstack((X_val.flatten()[:, None], T_val.flatten()[:, None]))
    u_val_star = Exact_val.flatten()[:, None]
    X_test_star = np.hstack((X_test.flatten()[:, None], T_test.flatten()[:, None]))
    u_test_star = Exact_test.flatten()[:, None]

    lb = X_tr_star.min(0)  
    ub = X_tr_star.max(0)

    xx1 = np.hstack((X_train[0:1, :].T, T_train[0:1, :].T))
    uu1 = Exact_train[0:1, :].T  
    xx2 = np.hstack((X_train[:, 0:1], T_train[:, 0:1])) 
    uu2 = Exact_train[:, 0:1] 

    X_u_train = np.vstack([xx1, xx2])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2])

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False) 
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    start_time = time.time()

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu, learning_rate, weighted)

    loss, loss_u, loss_f, mean_u_of_epoch, mean_f_of_epoch, mean_mape_u_of_epoch, var_u_of_epoch, var_f_of_epoch, var_mape_u_of_epoch, \
    worst_u_of_epoch, worst_f_of_epoch, worst_mape_u_of_epoch, epoch, error_u_extra, validation_error, best_it = \
        model.train(30000, 50, 0.00001, X_val_star, u_val_star, num_layers, num_neurons,learning_rate, weighted)
    u_pred_inter, f_pred_inter = model.predict(X_tr_star, best_it)
    u_pred_test, f_pred_test = model.predict(X_test_star, best_it)


    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    error_u_inter = np.linalg.norm(u_tr_star - u_pred_inter, 2) / np.linalg.norm(u_tr_star, 2)
    print('Error u: %e' % (error_u_inter))

    print('Error u: %e' % (error_u_extra))

    error_u_test = np.linalg.norm(u_test_star - u_pred_test, 2) / np.linalg.norm(u_test_star, 2)
    print('Error u: %e' % (error_u_test))

    return error_u_inter, error_u_extra, error_u_test, best_it



if __name__ == "__main__":

    num_layers = int(sys.argv[1])
    num_neurons = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    weighted = int(sys.argv[4])

    # num_layers = 8
    # num_neurons = 40
    # learning_rate = 0.0005
    # weighted = 1

    result = main_loop(num_layers, num_neurons, learning_rate, weighted)
    print(result)
