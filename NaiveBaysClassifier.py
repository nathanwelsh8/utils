class NaiveBaysianClassifier:
    """
        Naive Bayes Classifier
        
        Works by selecting the most likely class that each x belongs in. 
        Each class has a discriminant function which calculates this likelihood.

        Implemented by Nathan Welsh    
    """

    def __init__(self, verbose=False, use_log_probabilities = False):

        #prior probability for each class
        self.priors = np.array([0,0])
        self.verbose = verbose
        self.use_log_probabilities = use_log_probabilities

        # maps classes to numbers. This way classes
        # can be text or numbers 
        self.classes = {0:None, 1:None}
        self.input_classes = {}

        # feature averages per class
        self.feature_averages = None
        
        #init num perameters as 0
        self.num_features = 0

        # binary classification so 2 classes
        self.num_classes = 2

        # feature std per class
        self.feature_std = None

        # specifies the type of likelihood measure to use
        self.decision_function = {
            True:lambda X,_class:self.__log_likelihood(X,_class)+np.log(self.priors[_class]),
            False: lambda X,_class:self.__likelihood(X,_class)*self.priors[_class]
            }

    def __likelihood(self, X, _class):
        "returns the likelihood of each X beloning to the specified class"
        denom = 1/(np.sqrt(2*np.pi)*self.feature_std[_class])
        l_2_norm = (X-self.feature_averages[_class])**2

        exponential_term = np.exp(-(l_2_norm/(2*self.feature_std[_class]**2)))
        like = np.prod(denom*exponential_term, axis=1)
        return like

    def __log_likelihood(self,X,_class): 
        "returns the log-likelihood of each X beloning to the specified class "
        
        constant = -0.5*np.log(2*np.pi)
        l_2_norm = (X-self.feature_averages[_class])**2

        return np.sum(constant - 0.5*np.log(self.feature_std[_class]**2) -(1/(2*self.feature_std[_class]**2)) *l_2_norm, axis=1)

    def __calc_feature_means(self,X,y):

        # (2,N) dict containing feature means for each class
        self.feature_averages = np.zeros((self.num_classes, self.num_features))
        for i, _class in enumerate(self.classes):
            self.feature_averages[i] = np.mean(X[(y==i).ravel()], axis=0)
        
        if self.verbose:
            print("Feature averages:",self.feature_averages)

    def __calc_feature_std(self,X,y):

        # (2,N) dict containing feature std for each class
        self.feature_std = np.zeros((self.num_classes,self.num_features))
        for i, _class in enumerate(self.classes):
            self.feature_std[i] = np.std(X[(y==i).ravel()], axis=0)
        
        if self.verbose:
            print("Feature std:", self.feature_std)
            
    def __calc_prior(self, y):
        """
            Calculates the initial prior for each class to
            be its count within the sampled dataset.
        """
        classes, i = np.unique(y, return_counts=True)
        assert len(classes) == self.num_classes, f"More than {self.num_classes} classes detected"
        prior = i/len(y)

        if self.verbose:
            print("Calculated prior:", prior)
        return prior

    def __convert_class_to_0_1(self, y):
        """
            creates a class map so we can map from class name
            to a numerical representation.
            @param self.classes maps from {0,1}->class name
            @param self.input_classes maps from class name -> {0,1}  
        """
        detected_classes = np.unique(y)
        self.classes[0] = detected_classes[0]
        self.classes[1] = detected_classes[1]

        self.input_classes[detected_classes[0]] = 0
        self.input_classes[detected_classes[1]] = 1

        if self.verbose:
            print("Converted Classes:", self.classes)
            print("Input classes map:", self.input_classes)

    def __convert_class_arr_to_1_0(self, y):
        "maps y{c_1,c_2}->{0,1}"
        for i, _class in enumerate(y):
            y[i] = self.input_classes[_class[0]]
        return y
    
    def __to_numpy(self, d):
        "convert data to numpy array for processing"

        if type(d) == type(np.array([0])):
            return d
        if type(d) == type(pd.DataFrame()):
            return d.to_numpy()
        else:
            raise TypeError(f"{type(d)} is not supported, DataFrames or numpy arrays only!")

    def fit(self, X,y):
        X = self.__to_numpy(X)
        y = self.__to_numpy(y)
        self.num_features = X.shape[1]

        # setup class maps
        self.__convert_class_to_0_1(y)

        # map input classes to {0,1} classes
        y = self.__convert_class_arr_to_1_0(y)
        
        self.priors = self.__calc_prior(y)
        self.__calc_feature_means(X,y)
        self.__calc_feature_std(X,y)
        
        return self

    def transform(self, X,y):
        raise NotImplementedError("This method is not implemented")

    def predict(self, X):
        """
            Returns predicted class for an (X,N) samples.
            N must match the number of training features
        """
        assert X.shape[1] == self.num_features, f"X must contain {self.num_features} features."

        X = self.__to_numpy(X)
        class_proba = np.zeros((self.num_classes,X.shape[0]))
        for i in range(self.num_classes):
            class_proba[i] = self.decision_function[self.use_log_probabilities](X,i)
        
        # select class of most likelely
        pred_class = np.argmax(class_proba, axis=0)
        
        # convert back to original class name
        pred_class_labelled = [self.classes[x] for x in pred_class]
        return pred_class_labelled

        
