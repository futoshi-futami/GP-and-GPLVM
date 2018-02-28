from DGRFF_model_sup import Dgrff_model
import numpy as np

class Dgrff_opt:
    def __init__(self,N_tot,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff,train_set_x,train_label,Y_tr,batch_size,Y_validate,X_validate,Y_test,X_test,batch_size2):
                
        self.dgrff = Dgrff_model(N_tot,D,Q,Domain_number,Ydim,Hiddenlayerdim1,Hiddenlayerdim2,num_MC,n_rff)

        self.wrt = self.dgrff.wrt
        
        #self.dgrff.cal_check(train_set_x,Y_tr,train_label,batch_size)
        
        self.dgrff.lasagne_optimizer(train_set_x,Y_tr,train_label,batch_size)        
        self.dgrff.prediction_validation(Y_validate,X_validate,Y_test,X_test,batch_size2)

        self.train_model=self.dgrff.train_model
        self.test_model=self.dgrff.test_model
        self.validate_model=self.dgrff.validate_model
        self.f = self.dgrff.f