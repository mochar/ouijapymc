import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt
import arviz as az
from sklearn.decomposition import PCA


class ZeroInflatedStudentT(pm.Continuous):
    def __init__(self, nu, mu, sigma, pi, *args, **kwargs):
        super(ZeroInflatedStudentT, self).__init__(*args, **kwargs)
        self.nu = tt.as_tensor_variable(pm.floatX(nu))
        self.mu = tt.as_tensor_variable(mu)
        lam, sigma = pm.distributions.continuous.get_tau_sigma(tau=None, sigma=sigma)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.sigma = sigma = tt.as_tensor_variable(sigma)
        self.pi = pi = tt.as_tensor_variable(pm.floatX(pi))
        self.studentT = pm.StudentT.dist(nu, mu, lam=lam)
        
    def random(self, point=None, size=None):
        nu, mu, lam, pi = pm.distributions.draw_values(
            [self.nu, self.mu, self.lam, self.pi], 
            point=point, size=size)
        g = pm.distributions.generate_samples(
            sp.stats.t.rvs, nu, loc=mu, scale=lam**-0.5, 
            dist_shape=self.shape, size=size)
        return g * (np.random.random(g.shape) < (1 - pi))
    
    def logp(self, value):
        logp_val = tt.switch(
            tt.neq(value, 0),
            tt.log1p(-self.pi) + self.studentT.logp(value),
#             tt.log(self.pi))
            pm.logaddexp(tt.log(self.pi), tt.log1p(-self.pi) + self.studentT.logp(0)))
        return logp_val


class Ouija:
    def __init__(self, Y, response_type):
        self.Y = Y.apply(lambda x: x / np.mean(x[x > 0]), 1)
        self.N, self.P = self.Y.shape
        self.Y_switch = self.Y.loc[:, response_type == 'switch']
        _, self.P_switch = self.Y_switch.shape
        self.Y_transient = self.Y.loc[:, response_type == 'transient']
        _, self.P_transient = self.Y_transient.shape

        self._build_priors()
        self.epsilon = 0.01 # To avoid numerical issues
        self.lam = 10.0
        self.student_df = 10.0

        self._xs = np.linspace(0, 1, 100)

    def _build_priors(self):
        self.priors = {
            'pseudotime_means': np.repeat(0.5, self.N),
            'pseudotime_stds': np.repeat(1.0, self.N),

            'switch_strength_means': np.repeat(0.0, self.P_switch),
            'switch_strength_stds': np.repeat(5.0, self.P_switch),

            'switch_time_means': np.repeat(0.5, self.P_switch),
            'switch_time_stds': np.repeat(1.0, self.P_switch),

            'transient_length_means': np.repeat(50.0, self.P_transient),
            'transient_length_stds': np.repeat(10.0, self.P_transient),

            'transient_time_means': np.repeat(0.5, self.P_transient),
            'transient_time_stds': np.repeat(0.1, self.P_transient),
        }

    def _build_model(self, pseudotimes=None, testvals=None):
        '''
        Create the Ouija PyMC model.
        '''
        testvals = testvals if type(testvals) is dict else {}
        n_cells = self.N if pseudotimes is None else len(pseudotimes)
        
        with pm.Model() as model:
            # Pseudotimes
            if pseudotimes is None:
                t = pm.TruncatedNormal('t', 
                    self.priors['pseudotime_means'],
                    self.priors['pseudotime_stds'],
                    lower=0, upper=1, shape=self.N,
                    testval=testvals.get('t'))
            else:
                t = pm.Data('t', pseudotimes)
                
            #
            peak_hyper = pm.Uniform('peak_hyper', 0, 20)

            # Priors on switch
            peak_switch = pm.Gamma('peak_switch', peak_hyper / 2, 0.5, 
                shape=self.P_switch)
            strength_switch = pm.Normal('strength_switch',
                self.priors['switch_strength_means'], 
                self.priors['switch_strength_stds'],
                shape=self.P_switch)
            time_switch = pm.TruncatedNormal('time_switch',
                self.priors['switch_time_means'],
                self.priors['switch_time_stds'],
                lower=0, upper=1, shape=self.P_switch)

            # Priors on transient
            peak_transient = pm.Gamma('peak_transient', 
                peak_hyper / 2, 0.5, 
                shape=self.P_transient)
            length_transient = pm.TruncatedNormal('length_transient',
                self.priors['transient_length_means'],
                self.priors['transient_length_stds'],
                lower=0, shape=self.P_transient)
            time_transient = pm.TruncatedNormal('time_transient',
                self.priors['transient_time_means'],
                self.priors['transient_time_stds'],
                lower=0, upper=1, shape=self.P_transient)

            # Mean based on gene type
            mu_switch = pm.Deterministic('mu_switch', 2 * peak_switch / (1 + tt.exp(-1 * strength_switch * (tt.reshape(t, (n_cells, 1)) - time_switch))))
            mu_transient = pm.Deterministic('mu_transient', 2 * peak_transient * tt.exp(-1 * self.lam * length_transient * tt.square(tt.reshape(t, (n_cells, 1)) - time_transient)))

            # Std. based on mean-variance relationship
            phi = pm.Gamma('phi', 12, 4) # Overdispersion 
            std_switch = tt.sqrt((1 + phi) * mu_switch + self.epsilon)
            std_transient = tt.sqrt((1 + phi) * mu_transient + self.epsilon)

            # Dropout
            beta = pm.Normal('beta', 0, 0.1, shape=2)
            pi_switch = pm.math.invlogit(beta[0] + beta[1] * mu_switch)
            pi_transient = pm.math.invlogit(beta[0] + beta[1] * mu_transient)

            # Likelihood
            # mus = tt.concatenate([mu_switch, mu_transient], 1)
            # stds = tt.concatenate([std_switch, std_transient], 1)
            # pis = tt.concatenate([pi_switch, pi_transient], 1)
            # ZeroInflatedStudentT('obs', nu=self.student_df, 
            #     mu=mus, sigma=stds, pi=pis, shape=self.P,
            #     observed=np.c_[self.Y_switch, self.Y_transient][:n_cells, :])

            for p in range(self.P_switch):
                ZeroInflatedStudentT(self.Y_switch.columns[p], nu=self.student_df, 
                    mu=mu_switch[:, p], sigma=std_switch[:, p], pi=pi_switch[:, p], 
                    observed=self.Y_switch.iloc[:n_cells, p])

            for p in range(self.P_transient):
                ZeroInflatedStudentT(self.Y_transient.columns[p], nu=self.student_df,
                    mu=mu_transient[:, p], sigma=std_transient[:, p], pi=pi_transient[:, p], 
                    observed=self.Y_transient.iloc[:n_cells, p])  

        return model

    def _create_pp_model(self, pseudotimes):
        if not hasattr(self, 'pp_model') or self.pp_model is None:
            self.pp_model = self._build_model(pseudotimes=pseudotimes)
        else:
            with self.pp_model:
                pm.set_data({'t': pseudotimes})

    def sample_prior_predictive(self, xs=None, var_names=None):
        xs = self._xs if xs is None else xs
        self._create_pp_model(xs)
        with self.pp_model:
            return pm.sample_prior_predictive(var_names=var_names)

    def sample_post_predictive(self, xs=None, var_names=None):
        xs = self._xs if xs is None else xs
        self._create_pp_model(xs)
        with self.pp_model:
            #return pm.sample_posterior_predictive(self.trace, var_names=var_names)
            return pm.fast_sample_posterior_predictive(self.trace, var_names=var_names)

    def update_priors(self, updates):
        self.priors.update(updates)
        self.model = None
        self.pp_model = None

    def plot_prior_gene_curves(self, per_gene=False, n_curves=15):
        priorpred = self.sample_prior_predictive(var_names=['mu_switch', 'mu_transient'])
        sub = np.random.randint(0, 300, n_curves)

        fig, axs = plt.subplots(figsize=(10, 4), ncols=2, sharex=True)
        # axs[0].plot(self._xs, np.log1p(priorpred['mu_switch'][sub, :, 0].T), alpha=.5, c='r')
        axs[0].plot(self._xs, priorpred['mu_switch'][sub, :, 0].T, alpha=.5, c='r')
        axs[0].set_title('Switch-like')
        # axs[1].plot(self._xs, np.log1p(priorpred['mu_transient'][sub, :, 0].T), alpha=.5, c='r')
        axs[1].plot(self._xs, priorpred['mu_transient'][sub, :, 0].T, alpha=.5, c='r')
        axs[1].set_title('Transient')

        return fig

    def train(self, testvals=None, **sample_kwargs):
        self.model = self._build_model(testvals=testvals)
        sample_kw = dict(chains=4, draws=1000, tune=1000)
        sample_kw.update(sample_kwargs)
        with self.model:
            self.trace = pm.sample(**sample_kw)

    def save(self, out_dir='./trace/', overwrite=False):
        pm.save_trace(self.trace, out_dir, overwrite=overwrite)        

    def load(self, out_dir='./trace/'):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._build_model()
        self.trace = pm.load_trace(out_dir, self.model)

    def plot_overdispersion(self):
        var = (1 + self.trace['phi']) * self._xs[:, None] + self.epsilon

        fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.title('Overdispersion parameter (phi)')
        plt.hist(self.model['phi'].random(size=100), label='prior')
        plt.hist(self.trace['phi'], label='posterior')
        plt.legend()

        plt.subplot(122)
        plt.title('var = (1 + phi) * mu')
        plt.plot(self._xs, var.mean(1))
        az.plot_hpd(self._xs, var.T, ax=plt.gca())
        plt.xlabel('Mean')
        plt.ylabel('Variance')

        return fig

    def plot_dropout(self):
        xs = np.linspace(0, 3, 100)
        drop_pi = self.trace['beta'][:, 0] + self.trace['beta'][:, 1] * xs[:, None]

        fig, axs = plt.subplots(figsize=(10, 4), ncols=2)

        axs[0].set_title('Beta parameters')
        axs[0].hist(self.model['beta'].random(size=100)[:, 0], label='prior')
        axs[0].hist(self.trace['beta'][:, 0], label='post b0')
        axs[0].hist(self.trace['beta'][:, 1], label='post b1')
        axs[0].legend()

        axs[1].set_title('P(dropout) = inv-logit(b0 + b1 * mu)')
        axs[1].plot(xs, sp.special.expit(drop_pi).mean(1))
        az.plot_hpd(xs, sp.special.expit(drop_pi).T, ax=axs[1])

        fig.tight_layout()
        return fig

    def plot_gene_curves(self):
        postpred = self.sample_post_predictive(var_names=['mu_switch', 'mu_transient'])

        nrows = np.ceil(self.P / 2).astype(int)
        fig, axs = plt.subplots(figsize=(12, 13), ncols=2, nrows=nrows, sharex=True)
        axs_f = axs.flatten()

        for p in range(self.P_switch):
            ax = axs_f[p]
            # ax.scatter(self.trace['t'].mean(0), np.log1p(self.Y_switch.iloc[:, p]), alpha=0.5)
            # az.plot_hpd(self._xs, np.log1p(postpred['mu_switch'][:, :, p]), color='r', ax=ax)
            ax.scatter(self.trace['t'].mean(0), self.Y_switch.iloc[:, p], alpha=0.5)
            az.plot_hpd(self._xs, postpred['mu_switch'][:, :, p], color='r', ax=ax)
            ax.set_title(self.Y_switch.columns[p])
            
        for p in range(self.P_transient):
            ax = axs_f[self.P_switch + p]
            # ax.scatter(self.trace['t'].mean(0), np.log1p(self.Y_transient.iloc[:, p]), alpha=0.5)
            # az.plot_hpd(self._xs, np.log1p(postpred['mu_transient'][:, :, p]), color='r', ax=ax)
            ax.scatter(self.trace['t'].mean(0), self.Y_transient.iloc[:, p], alpha=0.5)
            az.plot_hpd(self._xs, postpred['mu_transient'][:, :, p], color='r', ax=ax)
            ax.set_title(self.Y_transient.columns[p])

        fig.tight_layout()        
        return fig

    def plot_curve_embedding(self, embedder, embedding, n_curves=15):
        '''
        TODO: Bin pseudotimes in segments so that colors are a bit more clear.
        '''
        postpred = self.sample_post_predictive(var_names=['mu_switch', 'mu_transient'])
        mus = np.concatenate((postpred['mu_switch'], postpred['mu_transient']), axis=2)
        mus = mus[:n_curves, :, :]

        # The ordering of the genes in the likelihood can be different than in Y, so we
        # need to set them in the correct order.
        indices = np.array([
            np.r_[self.Y_switch.columns, self.Y_transient.columns].tolist().index(col)
            for col in self.Y.columns])
        mus = mus[:, :, indices]

        # Map sampled means to embedding space
        mu_embedded = embedder.transform(mus.reshape(-1, self.P))

        # Plot gene curves in embedding
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], color='grey', alpha=.4)
        norm = plt.Normalize(self._xs.min(), self._xs.max())
        for i in range(n_curves):
            points = mu_embedded.reshape(n_curves, self._xs.shape[0], 2)[i, :, :].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(self._xs)
            lc.set_linewidth(2)
            plt.gca().add_collection(lc)

    def plot_predictive(self, embedder, embedding):
        xs = np.linspace(0.1, .9, 5)
        postpred = self.sample_post_predictive(xs, self.Y.columns)
        
        fig = plt.figure(figsize=(15, 3))
        gs = fig.add_gridspec(2, 5, height_ratios=[1, 20], hspace=.1, wspace=.05)

        for i, pseudotime in enumerate(xs):
            # Plot pseudotime value
            ax = fig.add_subplot(gs[0, i])
            ax.scatter(pseudotime, .5)
            ax.set_xlim(0, 1)
            ax.set_yticks([], minor=[])    
            ax.set_xticks([], minor=[])    
            ax.set_title(f'Pseudotime {pseudotime:.1f}')
            
            # Plot KDE of PCA
            Y_sampled = np.array([postpred[col][:, i] for col in self.Y.columns]).T
            sampled = embedder.transform(Y_sampled)
            
            ax = fig.add_subplot(gs[1, i])
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], color='grey', alpha=.3)
            sns.kdeplot(x=sampled[:, 0], y=sampled[:, 1], levels=5)
            ax.set_xlim(-2, 3)
            ax.set_ylim(-1.8, 2)
            ax.set_xticks([], minor=[])
            ax.set_yticks([], minor=[])
