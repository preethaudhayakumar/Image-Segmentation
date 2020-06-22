import sys
import numpy as np
from io_data import read_data,write_data
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class Expectation_Maximization:

#--------------------------------Function Initializaion---------------------------------

   def __init__(self,fname):

#Variable holders for data

       self.data = 0  		           
       self.image = 0
       self.X = 0

#Variable holders for probability related

       self.posterior = [[0,0]]	     
       self.prior = [0,0]
       self.normalization = [0,0]
       self.post_sum = [0,0]
       self.Mean = [[0,0,0],[0,0,0]]

       self.Cov1 = np.asarray([[0,0,0] , [0,0,0], [0,0,0]])
       self.Cov2 = np.asarray([[0,0,0] , [0,0,0], [0,0,0]])
       
#Preprocessing of data

       self.data,self.image = read_data(fname, True)
       self.X = self.data[:,2:]

#Initialization of means,variance
       self.Mean[0] = [self.X[0][0],self.X[0][1],self.X[0][2]]
       temp = [0,0,0]

       for i in self.X:  #finding the farthest value from cluster one and the same is assigned as cluster 2

           if temp[0] <  ((self.Mean[0][0] - i[0])  * (self.Mean[0][0] - i[0])):
               temp[0] = (self.Mean[0][0] - i[0])  * (self.Mean[0][0] - i[0])
               self.Mean[1][0] = i[0]

           if temp[1] <  ((self.Mean[0][1] - i[1])  * (self.Mean[0][1] - i[1])):
               temp[1] = (self.Mean[0][1] - i[1])  * (self.Mean[0][1] - i[1])
               self.Mean[1][1] = i[1]

           if temp[2] <  ((self.Mean[0][2] - i[2])  * (self.Mean[0][2] - i[2])):
               temp[2] = (self.Mean[0][2] - i[2])  * (self.Mean[0][2] - i[2])
               self.Mean[1][2] =  i[2]

       self.Cov1 = np.asarray([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
       self.Cov1 = np.cov(self.Cov1)       #covariance 1

       self.Cov2 = np.asarray([[15,0,0], [0, 15, 0], [0, 0, 15]])
       self.Cov2 = np.cov(self.Cov2)
       self.expect_maxim(fname)

#--------------------------------End of the function-------------------------------------


#--------------------------------Expectation and Maximization----------------------------
   
   def expect_maxim(self,filename):
       
#initializing for first iteration

       self.prior  = [0.5,0.5]
       exp_mean1 = 0
       exp_mean2 = 0
      
#Expectation step

       log_lh = []
       while(1):
          self.post_sum = [0,0]
          self.posterior = [[0,0]]
          exp_mean1 = [0,0,0]
          exp_mean2 = [0,0,0]

          for v in self.image:
            for i in v:
              l_post = [0,0]
              l_post[0] = multivariate_normal.pdf(i, mean= self.Mean[0], cov=self.Cov1, allow_singular=True)  # Likelihood 1
              l_post[1] = multivariate_normal.pdf(i, mean= self.Mean[1], cov=self.Cov2, allow_singular=True)  # Likelihood 2

              l_post[0] = self.prior[0] * l_post[0]
              l_post[1] = self.prior[1] * l_post[1]

              Normalization = l_post[0] + l_post[1]

              l_post[0] = l_post[0] / Normalization
              l_post[1] = l_post[1] / Normalization

              self.posterior.append(l_post)

              self.post_sum[0] = self.post_sum[0] + l_post[0]      #used in the maximization step
              self.post_sum[1] = self.post_sum[1] + l_post[1]
             
              exp_mean1 = exp_mean1 + (l_post[0] * i)
              exp_mean2 = exp_mean2 + (l_post[1] * i)

             
#Maximization Step :


          self.Mean[0] = exp_mean1 / self.post_sum[0]     #updated Mean
          self.Mean[1] = exp_mean2 / self.post_sum[1]

          variance1 = np.zeros((3, 3))
          variance2 = np.zeros((3, 3))
 
          j=1
          for v in self.image:
           for i in v:
              variance1 = variance1 + ( self.posterior[j][0] * np.outer((i-self.Mean[0]),(i-self.Mean[0])))
              variance2 = variance2 + ( self.posterior[j][1] * np.outer((i-self.Mean[1]),(i-self.Mean[1])))
              j = j+1
        
          self.Cov1 = variance1 / self.post_sum[0]  #updated variance
          self.Cov2 = variance2 / self.post_sum[1]

          self.prior[0] = self.post_sum[0] / self.X.shape[0]  #updated prior
          self.prior[1] = self.post_sum[1] / self.X.shape[0]
          print("Maximization ended")
          
          lval = 0
          sumList=[]
          
          for v in self.image:
           for i in v:
              l_post[0] = multivariate_normal.pdf(i, self.Mean[0], self.Cov1, allow_singular=True)
              l_post[1] = multivariate_normal.pdf(i, self.Mean[1], self.Cov2, allow_singular=True)

              Normalization = (self.prior[0] * l_post[0]) + (self.prior[1] * l_post[1])
              sumList.append(np.log(Normalization))

           lval = np.sum(np.asarray(sumList))

          log_lh.append(lval)
          print("Log Likelihood: " + str(lval))

          if len(log_lh) < 2: continue
          if np.abs(lval - log_lh[-2]) < 0.5: break
#end of While loop

#Copying mask

       backg = self.data.copy()
       foreg = self.data.copy()
       mask = self.data.copy()
              
       for i in range(0,len(self.data)-1):

          cell = self.data[i]
          point = [cell[2], cell[3], cell[4]]       
          l_post = [0,0]
          l_post[0] = multivariate_normal.pdf(point, mean= self.Mean[0], cov=self.Cov1, allow_singular=True)  # Likelihood 1
          l_post[1] = multivariate_normal.pdf(point, mean= self.Mean[1], cov=self.Cov2, allow_singular=True)  # Likelihood 2

          l_post[0] = self.prior[0] * l_post[0]
          l_post[1] = self.prior[1] * l_post[1]

          Normalization = l_post[0] + l_post[1]

          l_post[0] = l_post[0] / Normalization
          l_post[1] = l_post[1] / Normalization
          
          if (l_post[0] < l_post[1]):
              backg[i][2] = backg[i][3] = backg[i][4] = 0
              mask[i][2] = mask[i][3] = mask[i][4] = 0
          else:
              foreg[i][2] = foreg[i][3] = foreg[i][4] = 0
              mask[i][2] = 100
              mask[i][3] = mask[i][4] = 0

  
       write_data(backg,filename+"_back.txt")
       read_data(filename+"_back.txt", False, save=True, save_name=filename+"_background.jpg")

       write_data(foreg,filename+"_fore.txt")
       read_data(filename+"_fore.txt", False, save=True, save_name=filename+"_foreground.jpg")

       write_data(mask,filename+"_mask.txt")
       read_data(filename+"_mask.txt", False, save=True, save_name=filename+"_masked.jpg")

   
#--------------------------------End of the function-------------------------------------




if __name__ == "__main__":

    if len(sys.argv) != 2:
        print('Usage: python3 EM.py <filename.txt>')
        sys.exit(-1)
    Expectation_Maximization(sys.argv[1])
