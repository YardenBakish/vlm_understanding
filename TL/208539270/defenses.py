import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros([batch_size, *data_tr[0][0].shape], device = device)
    

    # total number of updates - FILL ME
    total_updates =  int(np.ceil(epochs / m))
    

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    count = 0
    for update_num in range(total_updates):
          for _, data in enumerate(loader_tr):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs.requires_grad = True
            for j in range(m):
                # zero the parameter gradients
     
                optimizer.zero_grad()
                # forward + backward + optimize - FILL ME
                #print(inputs.shape)
                #print(delta.shape)
                #print(delta[:inputs.shape[0]].shape)

                pred = model(inputs + delta[:inputs.shape[0]])
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

                delta[:inputs.shape[0]] +=  (eps * torch.sign(inputs.grad))
                delta = torch.clamp(delta,-eps,eps)
                
                count+=1
                if count % scheduler_step_iters == 0:
                    lr_scheduler.step()
    
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        # FILL ME
        x = x.squeeze(0)
        num_classes = 0
        with torch.no_grad():
            num_classes = self.model(x.unsqueeze(0)).shape[-1]
        
        hist = np.zeros(num_classes)
        remainder = n % batch_size
       
        for i in range(n // batch_size):
            noise = torch.randn(batch_size,*x.shape, device=x.device) *self.sigma
            pred = torch.argmax(self.model(x + noise), dim=-1)
            for c in pred:
                hist[c.item()]+=1
        if remainder > 0:
            noise = torch.randn(remainder, *x.shape, device=x.device) * self.sigma
            pred = torch.argmax(self.model(x + noise), dim=-1)
            for c in pred:
                hist[c.item()] += 1
        return hist
        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        c = (self._sample_under_noise(x,n0,batch_size)).argmax()
        
        
        # compute lower bound on p_c - FILL ME
        hist = self._sample_under_noise(x,n,batch_size)
        LB_pc = proportion_confint(hist[c],n,2*alpha,method="beta")[0]
        if LB_pc <= 0.5:
            return self.ABSTAIN, 0
        radius = norm.ppf(LB_pc)*self.sigma

        # done
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask    = torch.rand(*self.dim[2:],device=device)
        trigger = torch.rand(self.dim,device=device)
        mask.requires_grad = True
        trigger.requires_grad = True

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        for _ in range(max(1, int(np.ceil(self.niters / len(data_loader))))):
            for input, labels in data_loader:
                input = input.to(device)
               
                input_tag = (1-mask)*input+mask*trigger
                pred = self.model(input_tag)

                target = torch.full((input.shape[0],), c_t, device=device)

                pred_loss = self.loss_func(pred, target)
                mask_reg = (self.lambda_c * mask.abs().sum())
                loss = pred_loss+ mask_reg
                loss.backward()


                mask_step_size = self.step_size*mask.grad.sign()
                mask.data    = (mask- mask_step_size)
                trigger_step_size = self.step_size*trigger.grad.sign()
                trigger.data = (trigger- trigger_step_size)
                mask.data = mask.clamp(min=0,max=1)
                trigger.data = trigger.clamp(min=0,max=1)
              
                mask.grad = None
                trigger.grad = None
        mask = mask.repeat(1,3,1,1).to(device)

        # done
        return mask, trigger
