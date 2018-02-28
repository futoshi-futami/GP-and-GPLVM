from GP_model_reg import GP_model
import numpy as np

import sys,os
sys.path.append("./")
sys.path.append("../")
sys.path.append(os.pardir)

class GP_opt:
    def __init__(self,N_tot,D_in,D_out,M,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,train_set_x,Xlabel_share,Y_tr,batch_size,Y_validate,X_validate,Y_test,X_test,batch_size2):
        self.GP = GP_model(N_tot,D_in,D_out,M,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC)

        self.wrt = self.GP.wrt
        
        #self.GP.cal_check(train_set_x,Y_tr,Xlabel_share,batch_size)
        
        self.GP.lasagne_optimizer(train_set_x,Y_tr,Xlabel_share,batch_size)        
        self.GP.prediction_validation(Y_validate,X_validate,Y_test,X_test,batch_size2)

        self.train_model=self.GP.train_model
        self.test_model=self.GP.test_model
        self.validate_model=self.GP.validate_model
        self.f = self.GP.f