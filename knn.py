def knn(X, K, verbose=False):
    """"
        Performs KNN clustering on X 
    """
    # set K
    # Initialise the means
    mu = np.random.randn(K,2)
    # Set maximum number of iterations
    max_its = 200
    N = len(X)
    z = np.zeros((N,K))
    oldz = np.ones((N,K)) # just to make sure it is different from z in iteration 1
    # Colours for plotting - if you set K bigger than the 
    # length of this, it'll crash
    cols = ['ro','bo','go','yo','mo','ko']
    assert K <= len(cols), "There are more classes than class colours, add entries to cols"
    for it in range(max_its):
        
        # Assign each point to its closest mean vector
        # assignments are stored in z
        
        #calculate distance from X to mu's to 
        for i, m in enumerate(mu):
            z[:,i] = np.linalg.norm(X-m, axis=1)
        
        # assign x to the mu that it is closest too 
        mask = np.argmin(z, axis=1)
        for i, m in enumerate(mask):
            z[i,:] = np.zeros(K)
            z[i,:][m] =  1
            
        # Plot the status of the algorithm
        # The data are coloured according to the memberships in z
        # and the means are plotted in the same colours with larger symbols
        if(verbose):
            display(X,z,mu,K,cols, f"iteration {it}")
        
        
        # Check if anything has changes
        changes = (np.abs(z - oldz)).sum()
        if changes == 0:
            if verbose:
                print("Converged on iteration", it)
                display(X,z,mu,K,cols, "Final")
            break
        # Update the means  
        for k in range(K):
            mu[k] = np.mean(X[z[:,k]==1,:], axis=0)
            
        # Make a deep copy of z...
        oldz = np.copy(z)
    
    return np.argmax(z,axis=1)
    
assigned_classes = knn(X,K=3,verbose=True)
