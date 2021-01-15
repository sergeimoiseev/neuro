import numpy as np

class LogisticRegression:
    def __init__(self, fit_intercept=True):
        r"""
        A simple logistic regression model fit via gradient descent on the
        penalized negative log likelihood.

        Parameters
        ----------
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for `beta` will have `M + 1` dimensions,
            where the first dimension corresponds to the intercept. Default is
            True.
        """

        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7, method = 'sgd', hist_w = 0.9, batch_size = 10):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The binary targets for each of the `N` examples in `X`.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run the gradient descent
            solver. Default is 1e7.
        method : string
            The training method. Default is stochastic gradient descent - 'sgd'. 
            Formulas from https://remykarem.github.io/blog/gradient-descent-optimisers.html
            'sgd' for stochastic gradient descent
            'momentum' for Momentum,
            'nag' for Nesterov accelerated gradient,
            'rmsprop' for RMSProp.
        hist_w : float
            History weight for time-step methods.
        batch_size: int
            Size of subset of train data for 'sgd' method
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        l_prev = np.inf
        self.beta = np.random.rand(X.shape[1])

        # for momentum
        m_prev = np.zeros(X.shape[1])

        # for rms prop
        v_prev = np.zeros(X.shape[1])
        eps = 1e-6

        N, M = X.shape

        if method == 'sgd':
            for _ in range(int(max_iter)):
                rand_idx = np.random.randint(0,N)
                X_ = X[rand_idx:rand_idx+batch_size,:]
                y_ = y[rand_idx:rand_idx+batch_size]
                y_pred = sigmoid(np.dot(X_, self.beta))
                loss = -np.log(y_pred[y_ == 1]).sum() - np.log(1 - y_pred[y_ == 0]).sum()
                if np.abs(l_prev - loss) < tol:
                    return
                l_prev = loss
                grad = -(np.dot(y_ - y_pred, X_)) / N
                self.beta -= lr * grad 
        # TODO: ИМХО есть ошибка в реализации numpy_ml - во всех методах используется простой градиентный спуск, не sdg
        # и нужно по-хорошему перевести методы momentum, nag, rmsprop на основу sdg
        for _ in range(int(max_iter)):
            y_pred = sigmoid(np.dot(X, self.beta))
            loss = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
            if np.abs(l_prev - loss) < tol:
                return
            l_prev = loss
            grad = -(np.dot(y - y_pred, X)) / N
            if method=='gd':
                self.beta -= lr * grad 
            elif method=='momentum':
                m_prev = hist_w * m_prev + (1 - hist_w) * grad
                self.beta -= lr * m_prev
            elif method=='nag':
                prj_w = self.beta - lr * m_prev
                prj_y_pred = sigmoid(np.dot(X, prj_w))
                prj_grad = -(np.dot(y - prj_y_pred, X)) / N
                m_prev = hist_w * m_prev + (1 - hist_w) * prj_grad
                self.beta -= lr * m_prev
            elif method=='rmsprop':
                v_prev = hist_w * v_prev + (1 - hist_w) * np.square(grad)
                self.beta -= lr * grad / np.sqrt(v_prev + eps)
            else:
                pass

    # def update(self, X, y, y_pred):
    #     return

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z,)`
            The model prediction probabilities for the items in `X`.
        """
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(np.dot(X, self.beta))

def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))
