# Image-Segmentation

To Segment the given image as Foreground and Background using Expectation Maximization algorithm

Pseudocode
      
      1.  Initialize the parameters μ(mean),Σ(Co-variance),α(prior) Whereas, 
          
            Mean1 = First Pixel (L a b - format)
            Mean2 = Farthest pixel (L a b - format) from Mean1
          
            Co-Variance1 = Covariance([[20, 0 , 20] , [ 0, 20, 0] , [ 0, 0, 20]]) 
            Co-Variance2= Covariance([[15, 0 , 0 ] , [ 0,15, 0] , [ 0, 0, 15]])
          
            Prior (cluster 1) = 0.5 
            Prior (cluster 2) = 0.5
      
     2.   Followed by calculating the Expectation (calculation of posterior)
     
     3.   Calculation Maximisation step
            Updating Mean, Covariance, prior (assigning posterior to prior) respectively
            
     4.   Check for Convergence
     
     5.   If it is not converged repeat from step2
