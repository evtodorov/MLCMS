import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class MNIST:
    def __init__(self):
        """
        Wrapper class around loading the MNIST data to normalized between (0., 1.)
        """
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize
        self.X_train = X_train/255.
        self.Y_train = Y_train
        self.X_test  = X_test/255.
        self.Y_test  = Y_test

class VAE:
    def __init__(self, data_shape, latent_params, deep_layers = [256, 256], learning_rate=0.001, name=''):
        """
        Wrapper class for Variational Auto Encoder
        
        :param data_shape: (list) 
            Shape of the data to be processed by the model
        :param latent_params: (int)
            Size of the latent parameter space
        :param deep_layers: (list) defaults to [256, 256]
            Architechture of the hidden layers of the encoder and decoder (same for both).
            The number of entries determines the number of hidden layers. 
            The values are the size of each layer
        :param learning_rate: (float) default to 0.001
            Learning rate of the Adams optimizer used to minimize the ELBO
        :param name: str default to ''
            Additional naming identifier for the model
        :return: (object)
            Wrapper class
        """
        self.variance = tf.Variable([1.])
        self.data_shape = list(data_shape)
        self.learning_rate = learning_rate
        self.latent_params_size = latent_params
        self.deep_layers = deep_layers
        self.make_encoder = tf.make_template('encoder', self._make_encoder)
        self.make_decoder = tf.make_template('decoder', self._make_decoder)
        # model variables
        self.data_placeholder = tf.placeholder(tf.float32, [None] + self.data_shape)
        self.prior = self.make_prior(latent_params_size=self.latent_params_size)
        self.posterior = self.make_encoder(self.data_placeholder, latent_params_size=self.latent_params_size)
        self.latent_placeholder = self.posterior.sample()
        self.decoder = self.make_decoder(self.latent_placeholder, self.data_shape)
        self.likelihood = self.decoder.log_prob(self.data_placeholder)
        self.divergence = tfd.kl_divergence(self.posterior, self.prior)
        self.elbo = tf.reduce_mean(self.likelihood - self.divergence)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo)
        #prepare saver
        self.name = name
        self.saver = tf.train.Saver()
        self.save_path = f'./saved-models/VAE{self.name}_{self.data_shape}_{self.deep_layers}_{self.latent_params_size}'
        
    def make_prior(self, latent_params_size):
        """
        Create prior distribution p(z)

        :param latent_params_size: (int)
            Size of the latent parameter space
        :return: (tfd.Distribution)
        """

        means = tf.zeros(latent_params_size)
        stds = tf.ones(latent_params_size)
        return tfd.MultivariateNormalDiag(means, stds)

    def _make_encoder(self, x, latent_params_size):
        """
        Create encoder distribution q(z|x)

        :param x: (tf.placeholder)
            Placeholder for the input data to be encoded
        :param latent_params_size: (int)
            Size of the latent parameter space
        :return: (tfd.Distribution)
        """
        x = tf.layers.flatten(x)
        for layer in self.deep_layers:
            x = tf.layers.dense(x, layer, tf.nn.relu)
        means = tf.layers.dense(x, latent_params_size) # the mean can be any real number
        variances = tf.layers.dense(x, latent_params_size, tf.nn.softplus) # the variance needs to be positive
        return tfd.MultivariateNormalDiag(means, variances)

    def _make_decoder(self, z, data_shape):
        """
        Create decoder distribution p(x|z)

        :param z: (tf.placeholder)
            Placeholder for the latent space parameters to be decoded
        :param data_shape: (int)
            Shape of the data to be processed by the model
        :return: (tfd.Distribution)
        """
        flat_size = np.prod(data_shape)
        for layer in self.deep_layers:
            z = tf.layers.dense(z, layer, tf.nn.relu)
        means =  tf.reshape(tf.layers.dense(z, flat_size, tf.nn.sigmoid), [-1] + data_shape)
        variances = self.variance*tf.ones(data_shape)
        return tfd.Independent(tfd.MultivariateNormalDiag(means, variances),1)
        
    def plot_latent(self, codes, labels, epoch):
        """
        Plot the latent space

        :param codes: (np.array)
            Encodings in the latent space (z)
        :param labels: (np.array)
            Corresponding labels from the test set
        :param epoch: (int)
            Current epoch
        """
        plt.figure()
        ax = plt.gca()
        sc = ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, cmap='tab10')
        ax.set_aspect('equal')
        ax.set_xlim(codes.min() - .1, codes.max() + .1)
        ax.set_ylim(codes.min() - .1, codes.max() + .1)
        ax.tick_params(axis='both', which='both', left='off', bottom='off',
                        labelleft='off', labelbottom='off')
        plt.colorbar(sc, ticks=sorted(np.unique(labels)), label='digit')
        plt.title(f'Epoch {epoch}: Latent space')

    def plot_samples(self, fig, samples, ptype):
        """
        Plot samples from the prior

        :param fig: (plt.figure)
            Figure to plot on
        :param samples: (np.array)
            Array of samples to plot
        :param ptype: (str)
            Type of plotting to used
        """
        if ptype=='imshow':
            for index, sample in enumerate(samples,start=1):
                ax = fig.add_subplot(3,len(samples),index)
                ax.imshow(sample, cmap='gray')
                ax.axis('off')
        elif ptype=='scatter':
            ax = plt.subplot(131)
            ax.scatter(samples[:,0],samples[:,1])
            ax.set_aspect('equal')
        
    def plot_encode_decode(self, fig, inputs, outputs, ptype):
        """
        Plot inputs before encoding and after decoding 

        :param fig: (plt.figure)
            Figure to plot on
        :param inputs: (np.array)
            Array of inputs to plot
        :param outputs: (np.array)
            Array of outputs to plot
        :param ptype: (str)
            Type of plotting to used
        """
        if ptype=='imshow':
            for i, (inp, out) in enumerate(zip(inputs, outputs),start=1):
                axe = fig.add_subplot(3,len(inputs),len(inputs)+i)
                axe.imshow(inp, cmap='gray')
                axe.axis('off')
                axd = fig.add_subplot(3,len(outputs),2*len(outputs)+i)
                axd.imshow(out, cmap='gray')
                axd.axis('off')
        elif ptype=='scatter':
            ax = plt.subplot(132)
            ax.scatter(inputs[:,0],inputs[:,1])
            ax.set_aspect('equal')  
            ax = plt.subplot(133)
            ax.scatter(outputs[:,0],outputs[:,1])
            ax.set_aspect('equal') 
            
    def train(self, x_train, x_test, max_epochs, batch_size, threshold=0.001, plot_params = None):
        """
        Train the model and optionally display training diagonststics

        :param x_train: (np.array)
            Data to be used for training
        :param x_test: (np.array)
            Data to be used to evaluate perfromance 
        :param max_epochs: (int)
            Maximum epochs for the training
        :param batch_size: (int)
            Batch size of the training data
        :param threshold: (float), defaults to 0.001
            Ratio between consecutive evaluations of the ELBO on the test
            used to determine if training has converged
        :param plot_params: (dict) defaults to None
            If not None, training diagnostics will be displayed
            {
                'type': (str) 'imshow'|'scatter'
                'print_epochs': (list) List of epochs for which to show diagnostics
                'plot_sample': (int) Number of samples to plot
                'y_test': (np.array) Labels corresponding to x_test
            }
        """
        last_elbo = -1e-16
        nelbos = []
        if plot_params is not None:
            plot_latent_space = self.latent_params_size == 2
            try:
                ptype = plot_params['type'].strip().lower()
                print_epochs = sorted(plot_params['print_epochs'])
                n_samples = plot_params['plot_samples']
                y_test = plot_params['y_test']
            except KeyError as e:
                print("WARNING: Missing plot params: ", e)
                print("No visualisation will be shown")
                plot_params = None
            assert(print_epochs[-1]<=max_epochs)
            assert(ptype=='imshow' or ptype=='scatter')
            # visualisation variables
            encodings = self.posterior.mean()
            decoded = self.make_decoder(encodings,  self.data_shape).mean()
            samples = self.make_decoder(self.prior.sample(n_samples), self.data_shape).mean()
        init = tf.global_variables_initializer()
        with tf.train.MonitoredSession() as sess:
            sess.run(init)
            for epoch in range(max_epochs+1):
                feed = {self.data_placeholder: x_test}
                test_elbo = sess.run(self.elbo, feed)
                converged = (abs(test_elbo - last_elbo) < abs(threshold*test_elbo)) and (epoch > print_epochs[-1])
                nelbos.append(-test_elbo)
                last_elbo = test_elbo
                if (plot_params is not None) and (epoch in print_epochs or converged):
                    test_likelihood, test_divergence,\
                    test_latent, test_samples = sess.run([self.likelihood, self.divergence, 
                                                          self.latent_placeholder, samples], feed)
                    test_input = x_test[:n_samples]
                    test_decoded = sess.run(decoded, {self.data_placeholder: test_input})
                    print('Epoch', epoch, 'elbo', test_elbo,
                          'reconstruction',  np.mean(test_likelihood),
                          'divergence', np.mean(test_divergence))
                    if plot_latent_space:
                        self.plot_latent(test_latent, y_test, epoch)
                    if ptype=='imshow':
                        fig, bigax = plt.subplots(nrows=3, ncols=1, figsize=(n_samples, 3*1.27), sharey=True)
                        for i in range(3):
                            bigax[i].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
                            bigax[i]._frameon = False
                    else:
                        fig, bigax = plt.subplots(nrows=1, ncols=3, figsize=(20,5), sharey=True)
                    bigax[0].set_title('Epoch {}: generated'.format(epoch))
                    bigax[1].set_title('Epoch {}: input'.format(epoch))
                    bigax[2].set_title('Epoch {}: decoded'.format(epoch))
                    

                    self.plot_samples(fig, test_samples, ptype)
                    self.plot_encode_decode(fig, test_input, test_decoded, ptype)
                    plt.show()
                else:
                    print(test_elbo)
                if converged:
                    break
                for i in range(len(x_train)//batch_size):
                    feed = {self.data_placeholder: x_train[batch_size*i:batch_size*(i+1)]}
                    sess.run(self.optimize, feed)
                # shuffle the training  set
                perm = np.arange(len(x_train))
                np.random.shuffle(perm)
                x_train = x_train[perm]
            self.saver.save(sess._sess._sess._sess._sess,self.save_path)
        if plot_params is not None:
            plt.figure()
            plt.plot(nelbos)
            plt.xlabel('Epoch')
            plt.ylabel('ELBO')
            
    def passthrough(self, x_test):
        """
        Returns decoded images corresponding  to the encoded input

        :param x_test: (np.array)
            Input to encode and decode
        """
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            feed = {self.data_placeholder: x_test}
            encodings = self.posterior.mean()
            decoded = self.make_decoder(encodings,  self.data_shape).mean()
            test_decoded = sess.run(decoded, feed)
            return test_decoded
            
    def sample(self, n_samples):
        """
        Returns decoded images from random samples in the prior

        :param n_samples: (np.array)
            Number of samples
        """
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            samples = self.make_decoder(self.prior.sample(n_samples), self.data_shape).mean()
            return samples.eval()
