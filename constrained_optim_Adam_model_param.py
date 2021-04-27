import tensorflow as tf
import numpy as np

class ConstrainedOptAdamModel(object):
    def __init__(self, lr, opt, loss, delta, w):
        self.opt, self.lr, self.loss = opt, lr, loss
        self.w = w
        self.delta = delta
        

    def adapt_budget_penalty(self, objective, overbudget, variable):
        self.objective, self.overbudget, self.variable = objective, overbudget, variable
        self.grad_and_vars = self.opt.compute_gradients(self.loss)
        self.grad_and_vars_objective = self.opt.compute_gradients(self.objective)

        if self.variable is None:
            self.variable = tf.trainable_variables()
        
        return self._update_vector()

    def _update_vector(self):
        return tf.cond(self.overbudget>0.0, self._over_budget, self._set_2_one)

    def _update_delta(self):
        return tf.cond(self.ema_budget>0.0,self._over_delta,self._reduce_delta)

    def _over_delta(self):
        self.delta = self.w * self.delta
        
        return self.delta

    def _reduce_delta(self):
        self.delta= self.delta/self.w

        return self.delta
    
    def _over_budget(self):
        print("inoverbudget")
        new_list = []
        dos = self.opt.compute_gradients(self.objective, self.variable)
        dbs = self.opt.compute_gradients(self.overbudget, self.variable)

        do_grad = [do_grad[0] for do_grad in dos]
        connected = [db_grad[1] for db_grad in dbs ]

        db_grad = [db_grad[0] for db_grad in dbs ]
        db_grad_len = [(db_grad2[0].shape.as_list(), db_grad2[0], db_grad2[1]) for db_grad2 in dbs]
        grad_dict = {}
        
        for (g,v) in self.grad_and_vars:
            grad_dict[v] = g
        grads = [grad[0] for grad in self.grad_and_vars]

        do = tf.concat([tf.reshape(do, (-1,)) for do in do_grad], -1)
        db = tf.concat([tf.reshape(db, (-1,)) for db in db_grad], -1)
        grad = tf.concat([tf.reshape(grad, (-1,)) for grad in grads], -1)
        
        det2 = tf.reduce_sum(tf.multiply(db, grad))
        det3 = tf.reduce_sum(tf.multiply(db, do))
        det = tf.reduce_sum(tf.multiply(db, db))
              
        self.delta = self.delta * self.w
        all_v_b = db * ((det2 + self.delta) / det)
            
        s = 0
        for (length_array, gradient, variable) in db_grad_len:
            length = np.prod(length_array)
            e = s + length
            new_v = tf.reshape(all_v_b[s:e], length_array)
            new_list.append((tf.add(tf.where(tf.less(det3, 0.0), new_v, tf.zeros(shape=tf.shape(gradient))), grad_dict[variable]), variable))
            s += length
        
        return self.opt.apply_gradients(new_list)

    def _set_2_one(self):
        
        self.epsilon=self.delta/self.w
        return self.opt.apply_gradients(self.grad_and_vars_objective)


