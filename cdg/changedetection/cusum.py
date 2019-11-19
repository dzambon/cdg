# --------------------------------------------------------------------------------
# Copyright (c) 2017-2019, Daniele Zambon, All rights reserved.
#
# Implements several change-detection tests based on the cumulative sum chart.
# --------------------------------------------------------------------------------
import numpy as np
import scipy
import scipy.stats
import scipy.spatial.distance

import cdg.utils
import cdg.embedding
from cdg.changedetection import Cusum, WindowedAndMultivariateCusum


_gaussian_cusum_thresholds = {}
# _gaussian_cusum_thres[(beta, dof)]=[(arl0, threshold), (arl0, threshold), ...]
# 27/05/2018 -- simulation length = 2e4
_gaussian_cusum_thresholds[(0.75,1)] = [(1000, 16.197294106787684), (800, 15.5687661571645), (500, 13.88707729146491), (400, 13.167618065193729), (200, 10.45421450584535), (190, 10.266286049194285), (180, 9.967004847622622), (170, 9.733276832489254), (160, 9.5494228936201), (150, 9.250737388047511), (140, 8.969688454099742), (130, 8.756507093783336), (120, 8.468769016827853), (110, 8.139254444019477), (100, 7.851893474921201), (90, 7.556056717454974), (80, 7.18657821182719), (70, 6.688512128025678), (60, 6.080985524440666), (50, 5.618650509840411), (40, 4.90307864417124), (30, 4.130005572077784), (20, 3.1532768333623116), (10, 1.6865087425460241), (9, 1.4887172928998416), (8, 1.2990441092457579), (7, 1.0935028736426458), (6, 0.8577757838210878), (5, 0.6134002517128103), (4, 0.3023806223418297)]
_gaussian_cusum_thresholds[(0.75,2)] = [(1000, 16.976679577652828), (800, 16.230961043205912), (500, 14.46089213297008), (400, 13.652352304564964), (200, 11.705241338678302), (190, 11.367155008823762), (180, 11.068399281588107), (170, 10.949920954948078), (160, 10.770988148122012), (150, 10.454464833164621), (140, 10.24829821121805), (130, 9.870404724398377), (120, 9.611309286821152), (110, 9.438382152158148), (100, 9.062053809813975), (90, 8.525592089737566), (80, 8.209718824948794), (70, 7.8055551805465715), (60, 7.282667542274145), (50, 6.666465315624124), (40, 5.957787847071569), (30, 5.114117397217096), (20, 3.986475930106504), (10, 2.253361801463871), (9, 2.0188447172837014), (8, 1.7956139974691414), (7, 1.5130170021706641), (6, 1.1989630066607835), (5, 0.8587662546660582), (4, 0.4728929352794591)]
_gaussian_cusum_thresholds[(0.75,3)] = [(1000, 18.46290088397933), (800, 17.469947893254847), (500, 15.901187373245207), (400, 15.283517769933106), (200, 12.421216287520561), (190, 12.294978778964307), (180, 12.224770238957165), (170, 11.880063845367069), (160, 11.463481791895678), (150, 11.162080448187414), (140, 10.87832720639759), (130, 10.623258834151715), (120, 10.36980335047599), (110, 10.032916977394656), (100, 9.710406710167298), (90, 9.364145482801762), (80, 8.98750517863828), (70, 8.494956328708623), (60, 7.856523200343292), (50, 7.216676661336282), (40, 6.530687509776018), (30, 5.556792362297208), (20, 4.395464850225141), (10, 2.6093303824241376), (9, 2.3649925184241987), (8, 2.067517083887247), (7, 1.7452180582633652), (6, 1.3855019332893255), (5, 1.0192398621148424), (4, 0.555935337351575), (3, 0.024565659291885034)]
_gaussian_cusum_thresholds[(0.75,4)] = [(1000, 18.31992369391446), (800, 17.318476280204308), (500, 16.143257796713364), (400, 15.215381093932162), (200, 12.855762372362408), (190, 12.685944281266153), (180, 12.554960597164152), (170, 12.3633569517694), (160, 11.953096966441402), (150, 11.68619724385821), (140, 11.455860482374636), (130, 11.131538819893969), (120, 10.914984488113154), (110, 10.775701518330859), (100, 10.398583467696819), (90, 10.034304140627938), (80, 9.720326699439504), (70, 9.228053039115395), (60, 8.550573195139938), (50, 7.818853733998294), (40, 7.097586134319073), (30, 6.089332694026721), (20, 4.84234085723898), (10, 2.881058956056151), (9, 2.606125894452422), (8, 2.286518425643389), (7, 1.9299531992267367), (6, 1.5128439871240724), (5, 1.1170178689824457), (4, 0.5879404990685675), (3, 0.01145729885174429)]
_gaussian_cusum_thresholds[(0.75,5)] = [(1000, 21.140865766293388), (800, 20.344613577347438), (500, 18.566352300852564), (400, 17.88193971186645), (200, 14.255996571199832), (190, 14.128434250255786), (180, 13.734176169108643), (170, 13.530296927481608), (160, 13.31969023338332), (150, 12.970930444298244), (140, 12.60270658924537), (130, 12.346306262045644), (120, 11.953277906224692), (110, 11.586900840140885), (100, 11.153363126574416), (90, 10.713166577778814), (80, 10.215994135240393), (70, 9.740140554629203), (60, 9.058601217426276), (50, 8.414497841594617), (40, 7.712683675057115), (30, 6.718984688239204), (20, 5.359218696229827), (10, 3.204471767944608), (9, 2.8878319055889285), (8, 2.547823895481227), (7, 2.1811682598643314), (6, 1.7530199743879775), (5, 1.2399150767679856), (4, 0.6871262561601474)]
_gaussian_cusum_thresholds[(0.75,6)] = [(1000, 22.358034935353935), (800, 21.403268035785477), (500, 19.441568101083657), (400, 18.845136869375576), (200, 14.93113669637313), (190, 14.786077544523), (180, 14.545990473844554), (170, 14.309560034820585), (160, 14.017276091695841), (150, 13.64577644283413), (140, 13.287449272636774), (130, 13.065963061819946), (120, 12.731197496675694), (110, 12.08261054269748), (100, 11.75085412429095), (90, 11.295481875461817), (80, 10.780298023702397), (70, 10.267784532226319), (60, 9.638863684145623), (50, 8.9792710263048), (40, 8.226778221710166), (30, 7.176163768838883), (20, 5.728226353678251), (10, 3.4523373047492756), (9, 3.112266820670407), (8, 2.7365938925056463), (7, 2.347350870890396), (6, 1.8963456408906127), (5, 1.3482334240475327), (4, 0.7479978667219003), (3, 0.0021461672984930402)]
_gaussian_cusum_thresholds[(0.75,8)] = [(1000, 24.746131271808945), (800, 23.437908198705387), (500, 20.992530677996754), (400, 20.175074296605718), (200, 16.316090698923087), (190, 16.063776997486585), (180, 15.662339856023806), (170, 15.594452523310093), (160, 15.219852857708352), (150, 14.848002315302292), (140, 14.627466504142827), (130, 14.156325345473322), (120, 13.734831518536371), (110, 13.380975775937253), (100, 12.989639036385583), (90, 12.517261415245333), (80, 12.09547168687679), (70, 11.485851195104695), (60, 10.632635165667939), (50, 9.817964430245514), (40, 8.939664195249357), (30, 7.788548763502018), (20, 6.3328498340345725), (10, 3.813318757283918), (9, 3.4882544633041004), (8, 3.055985035200493), (7, 2.622854930100715), (6, 2.0889281339816304), (5, 1.5402648029566919), (4, 0.8615567691085761), (3, 0.017378946877853352)]
_gaussian_cusum_thresholds[(0.75,10)] = [(1000, 26.455656992042748), (800, 25.0410413132489), (500, 22.583748738465566), (400, 21.567634010511703), (200, 17.587040342930717), (190, 17.3082943899358), (180, 16.835363618790765), (170, 16.634128514784262), (160, 16.233732263265665), (150, 15.957125203101334), (140, 15.696523555707035), (130, 15.272206978747395), (120, 14.814442747215772), (110, 14.405294636149147), (100, 14.030184471318819), (90, 13.500282620491063), (80, 13.013426272449616), (70, 12.38628540054104), (60, 11.481810566601256), (50, 10.561598793291417), (40, 9.651426806076351), (30, 8.40780348983199), (20, 6.861717702608411), (10, 4.154768008926371), (9, 3.7948330231406384), (8, 3.3314436540971446), (7, 2.861544480035459), (6, 2.2823334552427514), (5, 1.6817675108014658), (4, 0.9361796929526296), (3, 0.011638681201093348)]
_gaussian_cusum_thresholds[(0.75,13)] = [(1000, 28.296360157359082), (800, 27.900281004676145), (500, 25.040144045846738), (400, 23.40371863403911), (200, 19.265145772217437), (190, 19.037670743130207), (180, 18.697548278960994), (170, 18.365886902804647), (160, 17.962153821555248), (150, 17.506475769936873), (140, 17.221704134955544), (130, 16.853655843986765), (120, 16.36560308125896), (110, 15.920911695324339), (100, 15.302897248322672), (90, 14.88305375800052), (80, 14.126384892435395), (70, 13.395868749927127), (60, 12.554408085228095), (50, 11.678338542139807), (40, 10.84058787312995), (30, 9.458138395587014), (20, 7.637662358216353), (10, 4.687179769448035), (9, 4.248599356243199), (8, 3.7305468994269706), (7, 3.214606179402425), (6, 2.624132297678605), (5, 1.8636912875878533), (4, 1.040197659069582), (3, 0.001800224762828151)]
_gaussian_cusum_thresholds[(0.75,15)] = [(1000, 30.512615098830874), (800, 28.75997470953908), (500, 26.31547443115484), (400, 24.60025216120133), (200, 20.349352600754386), (190, 20.16792859599866), (180, 19.54203865135308), (170, 19.129653540623547), (160, 18.732660989324174), (150, 18.509961331914592), (140, 18.091855102781057), (130, 17.619710557561675), (120, 17.20395448041212), (110, 16.7450340605394), (100, 16.305806322137148), (90, 15.699605665619671), (80, 15.109789223150536), (70, 14.37701160363509), (60, 13.417301095503763), (50, 12.34428564693405), (40, 11.262688802902275), (30, 9.804500477867368), (20, 8.066833766582523), (10, 4.914186494563435), (9, 4.4984053517947835), (8, 3.9425571704701348), (7, 3.394313860557155), (6, 2.7064630439303663), (5, 2.000714920728253), (4, 1.1202726158170222), (3, 0.016148104397441188)]




class GaussianCusum(WindowedAndMultivariateCusum):
    """
    Class implements `WindowedAndMultivariateCusum` for Gaussian processes.
    The cumulative sums are done over the sequence of squared Mahalanobis
    distances between the expectation of the Gaussian process and the observed
    vector.
    The `GaussianCusum` funded on the fact that the squared Mahalanobis distance
    is (proportional to) a chi^2(d) distribution.

    When the stream is windowed, it considers a stream of mean vectors (computed
    within each window). In this way, assuming sufficiently large window size,
    even when the original process is not Gaussian, the stream of sample means
    can be assumed to be a Gaussian one by the central limit theorem. Please
    refer to 
        Concept Drift and Anomaly Detection in Graph Streams
        Daniele Zambon, Cesare Alippi and Lorenzo Livi
        IEEE Transactions on Neural Networks and Learning Systems, 2018.
    for more details.

    Here, we assume that the covariance matrix doesn't change in swapping to a
    new distribution of the data.
    """

    def __init__(self, arl=100, window_size=1, beta=.75):
        super().__init__(arl=arl, window_size=window_size, beta=beta)
        self._training_sample_size = None
        self._mu_0 = None
        self._s2_0inv = None
        self._chi2dist = None

    def fit(self, x, estimate_threshold=True, **kwargs):
        """
        :param x: (no_train, d) training data.
        :param estimate_threshold: whether or not to estimate the threshold.
            Default is True.
        Optional parameters are the following:
            - mean: (d,) numpy array with the known expectation of the Gaussian
                process
            - inv_cov: (d, d) numpy array with the known inverse of the
                covariance matrix of the Gaussian process
            - cov:(d, d) numpy array with the known covariance matrix of the
                Gaussian process (if `inv_cov` is present, then `cov` is
                neglected)
            - verbose: passed
            - precompute_thresholds: passed
            - gamma_type: {'quantile', 'std', 'data'}
            - threshold_type: {'numerical', 'data'}
        :return: True if the procedure completed as expected, False otherwise
        """

        if x is None:
            self._training_sample_size = 1
        else:
            self._training_sample_size = x.shape[0]

        self._mu_0 = kwargs.pop('mean', None)
        if self._mu_0 is None:
            self._mu_0 = np.mean(x, axis=0)

        self._s2_0inv = kwargs.pop('inv_cov', None)
        if self._s2_0inv is None:
            s2_0 = kwargs.pop('cov', None)
            if s2_0 is None:
                s2_0 = np.cov(x.transpose())  # this is the unbiased version
            if s2_0.ndim == 0:
                self._s2_0inv = 1. / s2_0
            else:
                self._s2_0inv = np.linalg.inv(s2_0)

        self.sample_dim = x.shape[1]
        self._chi2dist = scipy.stats.chi2(df=self.sample_dim)

        res = super().fit(x=x, estimate_threshold=estimate_threshold, **kwargs)

        return res
    
    def _estimate_gamma(self, x, gamma_type='quantile', **kwargs):
        """
        Differently from its superclass method, gamma is taken by the
        the chi^2(d) distribution with d degrees of freedom.

        :param x: (no_train, d) training data.
        :param gamma_type: {'quantile', 'std', 'data'}
        """
        if gamma_type == "data":
            return super()._estimate_gamma(x, gamma_type='quantile')

        res = True
        if gamma_type == "quantile":
            gamma = self._chi2dist.ppf(self.beta)
        elif gamma_type == "std":
            gamma = self._chi2dist.mean() + self.beta * self._chi2dist.std()
        else:
            res = False
            raise ValueError(
                'gamma_type <{}> not available.'.format(str(gamma_type)))
        self.gamma = gamma
        return res

    def _estimate_threshold(self, x=None, threshold_type='numerical', **kwargs):
        """
        Differently from its superclass method, instead of estimating the
        threshold on a training set, it numerically computes it by running the
        super-method `_estimate_threshold` on a chi^2(d) simulated process.

        :param x: used only when threshold_type is 'data'
        :param threshold_type: {'numerical', 'data'} with `data` the threshold is estimated
            in a non-parameteric way, otherwise a gaussian is assumed
        :param kwargs: needs the required argument for 'numerical' threshold_type
            - len_simulation: length of the simulated chi^2(d) process
            - recompute_threshold: (False)
        :return: True/False
        """

        if threshold_type == "data":
            return super()._estimate_threshold(x)
        
        # if threshold is in table, then use it.
        current_setting = (self.beta, self._chi2dist.kwds['df'])
        threshold = None
        res = False
        # Whether to recompute thresholds
        recompute_threshold = kwargs.pop('recompute_threshold', False)
        if not recompute_threshold and current_setting in _gaussian_cusum_thresholds.keys():
            for e in _gaussian_cusum_thresholds[current_setting]:
                if e[0] == self.arl:
                    threshold = np.array([e[1]])
                    res = True

        # if threshold not is in table, estimate it
        if threshold is None:
            len_simulation = kwargs.pop('len_simulation', None)
            if len_simulation is None:
                len_simulation = 10 * self.arl
            self.log.info("estimating threshold...")
            plain_cusum = Cusum(arl=self.arl, beta=self.beta)
            plain_cusum.gamma = self.gamma
            d2_training = self._chi2dist.rvs(size=(int(len_simulation), 1))
            kwargs.pop('x', None)
            res = plain_cusum._estimate_threshold(x=d2_training, dof=self._chi2dist.kwds['df'],
                                                  **kwargs)
            threshold = plain_cusum.threshold
            
        # set threshold
        self.threshold = threshold
        return res

    @staticmethod
    def precomp_threshold(dof, len_sim=1e5, beta=.75):
        """
        Generates once for all certain common thresholds.
        :param dof: list of degrees of freedom.
        :param len_sim: length of the simulated sequence (default is 1e5).
        :param beta: sensitivity parameter (default is .75)
        """
        for d in dof:
            cdt = GaussianCusum(arl=None, beta=beta)
            cdt.fit(x=np.zeros((1, d)), estimate_threshold=True, len_simulation=len_sim,
                    verbose=True, precompute_thresholds=True)

    def compute_local_statistic(self, x_win):
        """
        Mahalanobis distance between the mean within the window `datapoint`, and
        the mean computed in the training phase.  More precisely, the distance
        is squared and rescaled to comply with a chi^2(d) distribution.

        :param x_win: (no_windows, window_size, d) windows of observations
        :return: squared and scaled Mahalanobis distance.
        """
        delta_mu = np.mean(x_win, axis=1) - self._mu_0.reshape(1, -1)
        # d2_ = np.sum(delta_mu.dot(self._s2_0inv) * delta_mu, axis=1)
        d2 = np.einsum('ij,ij->i', delta_mu.dot(self._s2_0inv), delta_mu)
        d2 /= (1. / self._training_sample_size + 1. / self.window_size)
        return d2

    @staticmethod
    def test_arl(d, arl, n):
        mu = np.random.randn(d)
        sigma = np.eye(d) + np.random.rand(d,d)
        # Sigma += Sigma.transpose()
        x = mu + np.dot(np.random.randn(n, d), sigma.transpose())
        cdt = cdg.changedetection.cusum.GaussianCusum(arl=arl)
        cdt.fit(x, estimate_threshold=True)
        cdt._mu_0 = mu
        cdt._s2_0inv = np.linalg.inv(np.dot(sigma, sigma.transpose()))
        y_predict, cumulative_sums = cdt.predict(x, reset=True, verbose=False)
        # plt.plot(y_predict)
        # plt.show()
        return np.mean(y_predict)


class ManifoldCLTCusum(GaussianCusum):
    """
    Extends `GaussianCusum` to work on a differential manifold. It is based on
    the Central Limit Theorem (CLT) on manifolds [3]; in particular, the CLT is
    computed on a tangent space.
    """

    def __init__(self, manifold, arl=100, window_size=1, beta=.75):
        parent_class = cdg.geometry.CCRiemannianManifold
        assert issubclass(type(manifold), parent_class)
        
        self.manifold = manifold
        super().__init__(arl=arl, window_size=window_size, beta=beta)
        self.manifold_mu0 = None

    def fit(self, x, estimate_threshold=False, **kwargs):
        """
        :param x: (no_train, emb_dim) training data. `emb_dim` is the dimension
            of the embedding representation of the points in the manifold (e.g.,
            usually is d+1, with d dimension of the manifold).
        :param estimate_threshold: whether or not to estimate the threshold.
            Default is False.
        Optional parameters are the following:
            - tangent_points: (no_train, d) numpy array with the local coordinates
                of x with respect to the mean in the tangent space ---which is 
                exactly (0, ..., 0).
            - tangent_inv_cov: (d, d) numpy array with the known inverse of the
                covariance matrix of the Gaussian process on the tangent space
        :return: True if the procedure completed as expected, False otherwise
        """
        Nu = kwargs.pop('tangent_points', None)
        if Nu is None:
            self.manifold_mu0 = self.manifold.sample_mean(X=x, radius=self.manifold.radius, **kwargs)
            Nu = self.manifold.log_map(x0_mat=self.manifold_mu0, X_mat=x)
        mahal_mat_inv = kwargs.pop('tangent_inv_cov', None)
        if mahal_mat_inv is None:
            Lambda, Sigma = self._manifold_matrices(Nu, self.manifold)
            Sigma_inv = np.linalg.inv(Sigma)
            mahal_mat_inv = np.dot(np.dot(Lambda, Sigma_inv), Lambda)
        return super().fit(x=Nu, estimate_threshold=estimate_threshold,
                           mean=Nu[0] * 0, inv_cov=mahal_mat_inv, **kwargs)

    @staticmethod
    def _manifold_matrices(Y, manifold):
        ntot, d = Y.shape

        if manifold.curvature == 0.:
            Sigma = np.dot(Y.transpose(), Y) / ntot
            return np.eye(d), Sigma

        sqrt_c = np.sqrt(np.abs(manifold.curvature))

        def f(t):
            return sqrt_c * t * np.cos(sqrt_c * t) / np.sin(sqrt_c * t)

        # Lambda_n
        Lambda = np.zeros((d, d))
        for n in range(ntot):
            y = Y[n:n + 1, :]
            # y_norm2 = manifold._geo.norm_squared(y)
            y_norm = np.linalg.norm(y)
            fy = f(y_norm)
            Lambda += 2 * (1 - fy) / (y_norm ** 2) * \
                np.dot(y.transpose(), y) + 2 * np.eye(d) * fy

        Lambda /= ntot

        # Sigma_tilde_n
        Sigma = 4 * np.dot(Y.transpose(), Y) / ntot
        return Lambda, Sigma

    def predict(self, x, reset=True, window_result=False, **kwargs):
        """
        Takes the embedding representation of the manifold testing data, and
        creates the corresponding stream on the tangent space by the Log map.
        """
        Nu = self.manifold.log_map(x0_mat=self.manifold_mu0, X_mat=x)
        return super().predict(x=Nu, reset=reset, window_result=window_result, **kwargs)


class DifferenceCusum(Cusum):
    """
    Extends `Cusum` by considering the monitoring a stream of scalar feature
    observations. It monitors the discrepancy of the observations from the mean
    observed in the training phase.

    It implements the structure for three sub-classes, according to three
    hypotheses encoded in the parameter `direction`
        - two-sided:  H_a : mu_1 neq mu_0
        - lower:      H_a : mu_1  <  mu_0
        - greater:    H_a : mu_1  >  mu_0
    which monitor respectively `abs(x-mu_0)`, `mu_0-x` and `x-mu_0`.
    """
    
    sample_dim = 1

    def __init__(self, arl=100, direction='greater', beta=.75):
        super().__init__(arl=arl, beta=beta)
        if direction not in ['greater', 'lower', 'two-sided']:
            raise ValueError(
                'direction <{}> not available.'.format(str(direction)))
        self.direction = direction
        self._mu_0 = None
        self._training_sample_size = None

    def fit(self, x, estimate_threshold=False, **kwargs):

        # if x is None:
        #     self._training_sample_size = 1
        # else:
        #     self._training_sample_size = x.shape[0]
        self._training_sample_size = x.shape[0]
        if x.ndim != 2:
            raise ValueError("x has to be (no_train, d) numpy array, but x.ndim = {}".format(x.ndim))
        elif x.shape[1] != 1:
            raise ValueError("{} can only deal with univariate data.".format(self.__class__))

        self._mu_0 = kwargs.pop('mean', None)
        if self._mu_0 is None:
            self._mu_0 = np.mean(x, axis=0)

        training_difference = self._process_data(x)
        res = super().fit(x=training_difference, estimate_threshold=estimate_threshold,
                          original_data=x, **kwargs)

        return res

    def iterate(self, datum):
        if datum.shape[0] > 1:
            raise NotImplementedError()
        difference = self._process_data(datum)
        return super().iterate(datum=difference)

    def _process_data(self, data):
        raise cdg.utils.AbstractMethodError()

    def _estimate_threshold(self, x, **kwargs):
        """
        :param x: (n, 1) training data processed by `_process_data`
        :param kwargs:
            - original_data : (n, 1) is the data without applying `_process_data`
        """
        ordata = kwargs.pop('original_data')
        return super()._estimate_threshold(ordata, **kwargs)

    @classmethod
    def test_arl(cls, arl, n_train, n_test):
        mu = np.random.randn()
        Sigma = np.random.rand()
        x = mu + np.random.randn(n_train + n_test, 1) * Sigma
        cdt = cls(arl=arl)
        cdt.fit(x[:n_train], estimate_threshold=True, verbose=True)
        y_predict, cumulative_sums = cdt.predict(x[n_train:], reset=True, verbose=False)
        return np.mean(y_predict)


class GreaterCusum(DifferenceCusum):
    def _process_data(self, data):
        data_processed = data.copy()
        data_processed -= self._mu_0
        return data_processed


class LowerCusum(DifferenceCusum):
    def _process_data(self, data):
        data_processed = data.copy()
        data_processed -= self._mu_0
        return - data_processed


class TwoSidedCusum(DifferenceCusum):
    def _process_data(self, data):
        data_processed = data.copy()
        data_processed -= self._mu_0
        return np.abs(data_processed)


class BonferroniCusum(Cusum):
    """
    Ensemble of change detection tests. The inference is made with Bonferroni correction.
    
    A general rule is that parameter x can be passed in two ways:
    :param x: it can be numpy array (no_train, d) and is applied to each
        cusum. Another possibility is to pass a list of no_cusum elements
        of dimensions (no_train, d_i) with d_i that may vary.

    """

    def __init__(self, cusum_list, arl=100):
        """

        :param cusum_list: a list of cusum already initialised
        :param arl: todo
        """
        super().__init__(arl=arl, beta=None)
        self.cusum_list = cusum_list
        self.no_cusum = len(cusum_list)
        for cusum in self.cusum_list:
            cusum.arl = int(arl * self.no_cusum)

    def fit(self, x, estimate_threshold=False, threshold_type='numerical', gamma_type='quantile', **kwargs):
        """
        Here the `fit` function is called on every cusum in the list.
        Parameters can be passed as list or single entities. If they are passed as lists,
        each element of the list is passed to the corresponding cusum in the list, otherwise
        the same value is passed to every cusum.
        :param x: it numpy array (no_train, d).
        :param estimate_threshold:  ... see Cusum
        :param threshold_type:  ... see Cusum
        :param gamma_type:  ... see Cusum
        :param kwargs:  ... see Cusum
        :return:
        """
        e_th = estimate_threshold if isinstance(estimate_threshold, list) else [estimate_threshold]*self.no_cusum
        th_t = threshold_type if isinstance(threshold_type, list) else [threshold_type]*self.no_cusum
        ga_t = gamma_type if isinstance(gamma_type, list) else [gamma_type]*self.no_cusum
        res = True
        for ci in range(self.no_cusum):
            cusum = self.cusum_list[ci]
            x_tmp = x[ci] if isinstance(x, list) else x
            res_tmp = cusum.fit(x=x_tmp, estimate_threshold=e_th[ci],
                                gamma_type=ga_t[ci], threshold_type=th_t[ci], **kwargs)
            res = res_tmp and res  # checks if every training procedure completed correctly
        return res

    def predict(self, x, reset=True, **kwargs):
        """
        :param x: it can be numpy array (no_train, d) and is applied to each
            cusum. Another possibility is to pass a list of no_cusum elements
            of dimensions (no_train, d_i) with d_i that may vary.
        ...
        """

        if isinstance(x, list):  # check if the I have to pass the same x to every cusum
            no_test = x[0].shape[0]
        else:
            no_test = x.shape[0]
        y_predict_list = np.empty((self.no_cusum, no_test))
        cumulative_sums_list = np.empty((no_test, self.no_cusum))
        for ci in range(self.no_cusum):
            cusum = self.cusum_list[ci]
            if isinstance(x, list):  # check if the I have to pass the same x to every cusum
                x_tmp = x[ci]
            else:
                x_tmp = x
            y_tmp, cum_tmp = cusum.predict(x=x_tmp, reset=reset, **kwargs)
            y_predict_list[ci] = y_tmp[:, 0]
            cumulative_sums_list[:, ci] = cum_tmp[:, 0]
        return np.max(y_predict_list, axis=0), cumulative_sums_list

    @staticmethod
    def test_arl(arl, n_train, n_test):
        mu = np.random.randn()
        Sigma = np.random.rand()
        x = mu + np.random.randn(n_train + n_test, 1) * Sigma
        cdt1 = cdg.changedetection.cusum.LowerCusum(arl=arl)
        cdt2 = cdg.changedetection.cusum.GreaterCusum(arl=arl)
        cdt = cdg.changedetection.cusum.BonferroniCusum(cusum_list=[cdt1, cdt2], arl=arl)
        cdt.fit(x[:n_train], estimate_threshold=True, verbose=True)
        y_predict, cumulative_sums = cdt.predict(x[n_train:], reset=True, verbose=False)
        return np.mean(y_predict)
    

def demo():
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    import cdg.embedding
    
    np.random.seed(123)
    # setup
    sample_size = 5000
    arl = 100
    win_size = 10
    cusum = []
    data_train = []
    data_test = []

    # create two multivariate distributions
    rv1 = multivariate_normal(mean=[0., 0.], cov=[[1., 0.], [0., .2]])
    rv2 = multivariate_normal(mean=[0., 0.2], cov=[[1., 0.], [0., 1.]])
    training_stream = rv1.rvs(size=1000)
    x1 = rv1.rvs(size=int(sample_size / 5 * 4))
    x2 = rv2.rvs(size=int(sample_size / 5))
    test_stream = np.concatenate((x1, x2), axis=0)
    
    # univariate
    training_stream_uni = training_stream[:, :1]
    test_stream_uni = test_stream[:, :1]

    # # euclidean data
    # man_euc = cdg.embedding.ccm.EuclideanManifold()
    # tmp = np.random.rand(1, 3) * 5  # wlog generate a mean
    # true_mean = man_tmp.clip(X_mat=tmp, radius=man_tmp.radius)
    # stream_euc_tr = man_tmp.exp_map(x0_mat=true_mean, Nu_mat=training_stream)
    # stream_euc_te = man_tmp.exp_map(x0_mat=true_mean, Nu_mat=test_stream)
    
    # spherical data
    man_sph = cdg.geometry.SphericalManifold(man_dim=2, radius=3)
    tmp = np.random.rand(1, 3) * 5  # wlog generate a mean
    true_mean = man_sph.clip(X_mat=tmp, radius=man_sph.radius)
    stream_sph_tr = man_sph.exp_map(x0_mat=true_mean, Nu_mat=training_stream)
    stream_sph_te = man_sph.exp_map(x0_mat=true_mean, Nu_mat=test_stream)

    # hyperbolic data
    man_hyp = cdg.geometry.HyperbolicManifold(man_dim=2, radius=3)
    tmp = np.random.rand(1, 3) * 5  # wlog generate a mean
    true_mean = man_hyp.clip(X_mat=tmp, radius=man_hyp.radius)
    stream_hyp_tr = man_hyp.exp_map(x0_mat=true_mean, Nu_mat=training_stream)
    stream_hyp_te = man_hyp.exp_map(x0_mat=true_mean, Nu_mat=test_stream)

    # gaussian no window
    cusum.append(GaussianCusum(arl=arl))
    data_train.append(training_stream)
    data_test.append(test_stream)
    # gaussian windowed
    cusum.append(GaussianCusum(arl=arl, window_size=win_size))
    data_train.append(training_stream)
    data_test.append(test_stream)
    
    for i in range(2):
        cusum.append(None)
        data_train.append(None)
        data_test.append(None)

    # lower
    cusum.append(LowerCusum(arl=arl))
    data_train.append(training_stream_uni)
    data_test.append(test_stream_uni)
    # greater
    cusum.append(GreaterCusum(arl=arl))
    data_train.append(training_stream_uni)
    data_test.append(test_stream_uni)
    # two-sided
    cusum.append(TwoSidedCusum(arl=arl))
    data_train.append(training_stream_uni)
    data_test.append(test_stream_uni)
    # bonferroni on different cusum
    bonf_cusum = BonferroniCusum(arl=arl, cusum_list=[LowerCusum(arl=arl),
                                                      TwoSidedCusum(arl=arl)]) #,
                                                      # GreaterCusum(arl=arl)])
    cusum.append(bonf_cusum)
    data_train.append(training_stream_uni)
    data_test.append(test_stream_uni)

    # euclidean windowed
    # cusum_euc = ManifoldCLTCusum(arl=arl, manifold=man_euc, window_size=win_size)
    cusum_euc = GaussianCusum(arl=arl, window_size=win_size)
    cusum.append(cusum_euc)
    data_train.append(training_stream)
    data_test.append(test_stream)
    # spherica windowed
    cusum_sph = ManifoldCLTCusum(arl=arl, manifold=man_sph, window_size=win_size)
    cusum.append(cusum_sph)
    data_train.append(stream_sph_tr)
    data_test.append(stream_sph_te)
    # hyperbolic windowed
    cusum_hyp = ManifoldCLTCusum(arl=arl, manifold=man_hyp, window_size=win_size)
    cusum.append(cusum_hyp)
    data_train.append(stream_hyp_tr)
    data_test.append(stream_hyp_te)
    # bonferroni on different cusum
    bonf_cusum = BonferroniCusum(arl=arl, cusum_list=[cusum_euc, cusum_sph, cusum_hyp])
    cusum.append(bonf_cusum)
    data_train.append([training_stream, stream_sph_tr, stream_hyp_tr])
    data_test.append([test_stream, stream_sph_te, stream_hyp_te])

   

    fig1 = plt.figure()
    for ci in range(len(cusum)):
        if not cusum[ci] is None:
            cusum[ci].fit(data_train[ci], estimate_threshold=True, len_simulation=1000)
            y_pred, gg = cusum[ci].predict(data_test[ci], reset=False)
            gg = np.mean(gg, axis=1) # only necessary for bonferroni
            sp = fig1.add_subplot(3,4, 1 + ci)
            sp.plot(y_pred*max(gg), '+k')
            sp.plot(gg, label='g')
            sp.plot([cusum[ci].threshold] * len(gg), label='h')
            sp.grid(True)
            sp.set_title(str(type(cusum[ci]))[-20:])

    plt.show()

def test1():
    
    x = np.random.uniform(100, 500, size=(10000, 3))
    x_test = np.random.uniform(100, 500, size=(200000, 3))

    
    cdt = GaussianCusum(arl=100, window_size=100)
    cdt.fit(x, estimate_threshold=True, len_simulation=1e3)

    pred, cum_sum = cdt.predict(x_test, reset=True)
    pred = np.array(pred).astype(int)

    y_true = np.zeros((1,20000))
    y_pred = pred
    # y_pred = pred.reshape(-1,1000).mean(-1).round().reshape(-1)

    import matplotlib.pyplot as plt
    plt.plot(cum_sum)
    plt.axhline(cdt.threshold)
    plt.show()
    from sklearn.metrics import confusion_matrix
    # # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # tpr = tp / (tp + fn)
    # fpr = fp / (fp + tn)
    return


def test2():
    import numpy as np
    N = 400
    N_train = 100
    N_change = 320
    alpha = 0.01
    
    x = np.random.normal(size=(N, 1))
    x[N_change:] += 1.
    
    from cdg.changedetection import GaussianCusum
    cdt = GaussianCusum(arl=round(1. / alpha))
    cdt.fit(x[:N_train])
    y, g = cdt.predict(x, reset=False)

    cdt.reset()
    print(cdt.threshold)
    cdt.fit(x[:N_train])
    for t in range(N):
        alarm, _ = cdt.iterate(x[t:t + 1])
        if alarm:
            print("An alarm is raised at time {}".format(t))
            cdt.reset()
    
    
def test3():
    from cdg.graph import DelaunayGraphs, convert
    no_nodes = 5
    no_graphs = {0: 500, 8: 50}
    model = DelaunayGraphs()
    G = model.get(seed_points=no_nodes, classes=list(no_graphs.keys()),
                  no_graphs=no_graphs, sigma=.3, include_seed_graph=False)
    
    from cdg.graph.distance import GraphEditDistanceNX
    ged = GraphEditDistanceNX(node_cost='euclidean', n_jobs=2)
    Gnx = convert(G[0] + G[8], format_in='cdg', format_out='nx')
    G_train, G_test = Gnx[:50], Gnx[50:]
    
    from cdg.embedding import MultiDimensionalScaling
    mds = MultiDimensionalScaling(emb_dim=2, nprot=5)
    mds.fit(graphs=G_train, dist_fun=ged.get_measure_fun(verbose=True))
    x = mds.transform(G_test)
    
    from cdg.changedetection import GaussianCusum
    cdt = GaussianCusum(window_size=5, arl=20)
    cdt.fit(x[:100])
    y, g = cdt.predict(x, reset=False)

if __name__ == "__main__":
    # GaussianCusum.precomp_threshold([1, 2], len_sim=1e3)
    # test1()
    # demo()
    # test2()
    test3()
