import sys
sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from constrained_optim_Adam_model_param import ConstrainedOptAdamModel
import os
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error

np.random.seed(1234)
tf.set_random_seed(1234)

class DPM:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, learning_rate, epsilon, delta, w):

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
        self.epsilon = epsilon
        self.delta = delta
        self.w = w 
        
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

        
        self.loss_u = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)+0.01*tf.nn.l2_loss(self.W_reg)) 
        self.loss_f = tf.reduce_mean(tf.square(self.f_pred)+0.01*tf.nn.l2_loss(self.W_reg))

        
        self.partial_loss = self.loss_u + tf.nn.relu(self.loss_f - self.epsilon)
        self.new_loss = self.partial_loss
        self.norelu_loss = self.loss_u + self.loss_f
        
        self.mape_u = tf.reduce_mean(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))
        
        self.var_u = tf.math.reduce_variance(tf.square(self.u_tf - self.u_pred))
        self.var_f = tf.math.reduce_variance(tf.square(self.f_pred))
        self.var_mape_u = tf.math.reduce_variance(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))

        self.worst_u = tf.math.reduce_max(tf.square(self.u_tf - self.u_pred))
        self.worst_f = tf.math.reduce_max(tf.square(self.f_pred))
        self.worst_mape_u = tf.reduce_max(tf.abs(self.u_tf - self.u_pred) / tf.abs(self.u_tf))
        self.opt = tf.train.AdamOptimizer(learning_rate= self.learning_rate)
        
        self.com = ConstrainedOptAdamModel(lr=self.learning_rate, opt=self.opt, loss=self.norelu_loss,
                                           delta=self.delta, w = self.w)
        self.update_Adam = self.com.adapt_budget_penalty(self.loss_u, self.loss_f - self.epsilon, None)
        
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
        self.W_reg = weights[0]
        b = biases[0]
        H = tf.tanh(tf.add(tf.matmul(H, self.W_reg), b))
        for l in range(1, num_layers - 2):
            self.W_reg = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.add(tf.matmul(H, self.W_reg), b), H))
        self.W_reg = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, self.W_reg), b)

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


    def train(self, nIter, n_outlook, diff_loss, X_val_star, u_val_star, num_layers, num_neurons, learning_rate, epsilon, delta, w):

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
            _, new_loss, lossu, lossf,_com_overbudget= self.sess.run([self.update_Adam, self.norelu_loss, self.loss_u, self.loss_f,self.com.overbudget], tf_dict)
            
            loss_of_all.insert(it, self.sess.run(self.new_loss, tf_dict))
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

            if it % n_outlook == 0:
                self.saver.save(self.sess,
                                './tf_model/Viscous_burgers-con_limit-ResNet-DPM-%d-%d-%.4f-%.4f-%.3f-%.3f-%d.ckpt' % (
                               num_layers, num_neurons, learning_rate, epsilon, delta, w ,it))

                u_tf_pred, f_tf_pred = self.predict_val(X_val_star)
                val_error = np.linalg.norm(u_val_star - u_tf_pred, 2) / np.linalg.norm(u_val_star, 2)
                val_err.insert(it, val_error)

                if best_val > val_error:
                    best_val = val_error
                    i = it
                    self.saver.save(self.sess,'./tf_model/Viscous_burgers-con_limit-ResNet-DPM-%d-%d-%.4f-%.4f-%.3f-%.3f.ckpt' % (
                                                            num_layers, num_neurons, learning_rate, epsilon, delta, w))
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f over_budget : %.3f' % (it, new_loss, elapsed,_com_overbudget))
                start_time = time.time()

        return loss_of_all, loss_of_u, loss_of_f, mean_of_epoch_u, mean_of_epoch_f, mean_of_mape_u, var_of_epoch_u, var_of_epoch_f, \
               var_of_mape_u, worst_of_epoch_u, worst_of_epoch_f, worst_of_mape_u, n_epoch, best_val, val_err, i

    def predict_val(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star

    def predict(self, X_star, it):
        self.saver.restore(self.sess,
                           './tf_model/Viscous_burgers-con_limit-ResNet-DPM-%d-%d-%.4f-%.4f-%.3f-%.3f-%d.ckpt' % (
                               num_layers, num_neurons, learning_rate, epsilon, delta, w, it))

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star

    def plotting_whole_graph(self, U_star, U_pred, X_u_train, u_train, x, t, Exact):
        
        fig, ax = newfig(1.0, 0.5)
        ax.axis('off')
        
        ####### Row 0: ground truth u(t,x) ##################
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=4.75, bottom=4.15, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        
        h = ax.imshow(U_star.T, interpolation='nearest', cmap='rainbow',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        
        ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
                clip_on=False)
        
        line = np.linspace(x.min(), x.max(), 2)[:, None]
        ax.plot(t[:, 1000] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1372] * np.ones((2, 1)), line, 'w-', linewidth=1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.legend(frameon=False, loc='best')
        ax.set_title('$u(t,x)$', fontsize=10)
        
        ####### Row 1: prediction u(t,x) ##################
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=3.75, bottom=3.15, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs1[:, :])
        
        h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                      extent=[t.min(), t.max(), x.min(), x.max()],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        
        ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
                clip_on=False)
        
        line = np.linspace(x.min(), x.max(), 2)[:, None]
        ax.plot(t[:, 172] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 515] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 800] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1007] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1149] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1372] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1429] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1606] * np.ones((2, 1)), line, 'w-', linewidth=1)
        ax.plot(t[:, 1972] * np.ones((2, 1)), line, 'w-', linewidth=1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.legend(frameon=False, loc='best')
        ax.set_title('$u(t,x)$', fontsize=10)
        
        ####### Row 2: training set interpolation ##################
        
        gs2 = gridspec.GridSpec(1, 3)
        gs2.update(top=2.75, bottom=2.1, left=0.1, right=0.9, wspace=0.5)
        
        ax = plt.subplot(gs2[:, 0])
        ax.plot(x.transpose(), Exact[172, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[172, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 2.9925$', fontsize=10)
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        
        ax = plt.subplot(gs2[:, 1])
        ax.plot(x.transpose(), Exact[515, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[515, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 8.995$', fontsize=10)
        
        ax = plt.subplot(gs2[:, 2])
        ax.plot(x.transpose(), Exact[800, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[800, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 13.9825$', fontsize=10)

        ####### Row 3: training set interpolation ##################
        gs3 = gridspec.GridSpec(1, 3)
        gs3.update(top=1.7, bottom=1.05, left=0.1, right=0.9, wspace=0.5)
        
        ax = plt.subplot(gs3[:, 0])
        ax.plot(x.transpose(), Exact[1007, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1007, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 17.605$', fontsize=10)
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        
        ax = plt.subplot(gs3[:, 1])
        ax.plot(x.transpose(), Exact[1149, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1149, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 20.09$', fontsize=10)
        
        ax = plt.subplot(gs3[:, 2])
        ax.plot(x.transpose(), Exact[1372, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1372, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 23.9925$', fontsize=10)

        ####### Row 4: training set interpolation ##################
        gs4 = gridspec.GridSpec(1, 3)
        gs4.update(top=0.65, bottom=0, left=0.1, right=0.9, wspace=0.5)
        ax = plt.subplot(gs4[:, 0])
        ax.plot(x.transpose(), Exact[1429, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1429, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_title('$t = 24.99$', fontsize=10)
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        
        ax = plt.subplot(gs4[:, 1])
        ax.plot(x.transpose(), Exact[1606, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1606, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 28.0875$', fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
        
        ax = plt.subplot(gs4[:, 2])
        ax.plot(x.transpose(), Exact[1972, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x.transpose(), U_pred[1972, :], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')
        ax.set_xlim([-5, 105])
        ax.set_ylim([0, 6])
        ax.set_title('$t = 34.4925$', fontsize=10)
        
        # savefig('./figures/Viscous_burgers-DPM-%d-%d-%.4f-%.3f-%.2f-%.3f' % (num_layers, num_neurons, learning_rate, epsilon, delta, w))


def main_loop(num_layers, num_neurons, learning_rate, epsilon, delta, w):

    N_u = 100
    N_f = 10000

    nu = 0.01 / np.pi

    num_layers = num_layers
    num_neurons = num_neurons
    learning_rate = learning_rate
    epsilon = epsilon
    delta = delta 
    w = w 
    
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

    model = DPM(X_u_train, u_train, X_f_train, layers, lb, ub, nu, learning_rate, epsilon, delta, w)

    loss, loss_u, loss_f, mean_u_of_epoch, mean_f_of_epoch, mean_mape_u_of_epoch, var_u_of_epoch, var_f_of_epoch, var_mape_u_of_epoch, \
    worst_u_of_epoch, worst_f_of_epoch, worst_mape_u_of_epoch, epoch, error_u_extra, validation_error, best_it =\
        model.train(30000, 50, 0.00001, X_val_star, u_val_star, num_layers, num_neurons, learning_rate, epsilon, delta, w)
    u_pred_inter, f_pred_inter = model.predict(X_tr_star, best_it)
    u_pred_test, f_pred_test = model.predict(X_test_star, best_it)
    u_pred, f_pred = model.predict(X_star, best_it)

    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    error_u_inter = np.linalg.norm(u_tr_star- u_pred_inter, 2) / np.linalg.norm(u_tr_star, 2)
    print('Error u: %e' % (error_u_inter))

    print('Error u: %e' % (error_u_extra))

    error_u_test = np.linalg.norm(u_test_star - u_pred_test, 2) / np.linalg.norm(u_test_star, 2)
    print('Error u: %e' % (error_u_test))

    # plotting

    U_star = griddata(X_star, u_star.flatten(), (X, T), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
 
    model.plotting_whole_graph(U_star, U_pred, X_u_train, u_train, x, t, Exact)

    return error_u_inter, error_u_extra, error_u_test, best_it


if __name__ == "__main__":

    num_layers = int(sys.argv[1])
    num_neurons = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    epsilon = float(sys.argv[4])
    delta = float(sys.argv[5])
    w = float(sys.argv[6])
    
    result = main_loop(num_layers, num_neurons, learning_rate, epsilon, delta, w)
    print(result)
    
