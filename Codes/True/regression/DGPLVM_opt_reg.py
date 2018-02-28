from DGPLVM_model_reg import DGGPLVM_model
import numpy as np

class DGGPLVM_opt:
    def __init__(self,N_tot,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff,train_set_x,train_label,Y_tr,batch_size,Y_validate,X_validate,Y_test,X_test,batch_size2):
        self.DGPLVM = DGGPLVM_model(N_tot,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff)

        self.wrt = self.DGPLVM.wrt
        
        #self.DGPLVM.cal_check(train_set_x,Y_tr,train_label,batch_size)
        
        self.DGPLVM.lasagne_optimizer(train_set_x,Y_tr,train_label,batch_size)        
        self.DGPLVM.prediction_validation(Y_validate,X_validate,Y_test,X_test,batch_size2)

        #self.train_model=self.DGPLVM.train_model
        self.test_model=self.DGPLVM.test_model
        self.validate_model=self.DGPLVM.validate_model
        self.f = self.DGPLVM.f