import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        x_tag = x.clone()
        if self.rand_init:
            x_tag+=torch.distributions.Uniform(-self.eps, self.eps).sample(x_tag.size()).to(x_tag.device)
        for i in range(self.n):
            self.model.zero_grad()
            x_tag.requires_grad = True
            logits              = self.model(x_tag)
            if self.early_stop:
                mask = (torch.argmax(logits,dim=-1) == y)
                num_correct = mask.sum()
                if ((targeted==True) and (num_correct==y.shape[0])): 
                    break
                if ((targeted==False) and (num_correct == 0)):
                    break
                if targeted:
                    mask = ~mask
            
            loss  = self.loss_func(logits,y).mean()
            loss.backward(retain_graph=True)
     
            with torch.no_grad():
                  gradients = x_tag.grad.sign()
                  if self.early_stop:
                      #zero-down gradients if corresponding input reached goal
                      gradients*= mask.view(-1, 1, 1, 1)
                  elem = -self.alpha*gradients if targeted else self.alpha*gradients
                  x_tag+=elem
                  x_tag = x_tag.clamp(min=0, max=1)
                  x_tag = x_tag.clamp(min=x-self.eps, max=x+self.eps)
                  
        return x_tag


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        grads     = torch.zeros_like(x)
        q_num_arr = torch.zeros_like(y)
        x_tag = x.clone()
        if self.rand_init:
            x_tag+=torch.distributions.Uniform(-self.eps, self.eps).sample(x_tag.size()).to(x_tag.device)
        
        with torch.no_grad():
            for i in range(self.n):
                  mask = (torch.argmax(self.model(x_tag),dim=-1)  == y)
                  if self.early_stop:
                        num_correct = mask.sum()
                        if ((targeted==True) and (num_correct==y.shape[0])): 
                            break
                        if ((targeted==False) and (num_correct == 0)):
                            break
                  if targeted:
                    mask = ~mask
                  q_num_arr+=(mask*(2 * self.k  ) )
                  ####NES########################
                  g = torch.zeros_like(x_tag)
                  for j in range(self.k):
                      u_j = torch.randn_like(x_tag)
                      p_plus  = self.loss_func(self.model(x_tag+self.sigma*u_j), y).view(-1, 1, 1, 1)*u_j
                      g+=p_plus
                      p_neg = self.loss_func(self.model(x_tag-self.sigma*u_j), y).view(-1, 1, 1, 1)*u_j
                      g-=p_neg
                  nes = ((2*self.k*self.sigma)**-1) * g
                  ###################
                  grads = self.momentum*grads + (1-self.momentum)*nes
                  gradients = torch.sign(grads)
                  if self.early_stop:
                      gradients*= mask.view(-1, 1, 1, 1)
                  if targeted:
                      gradients = -gradients
                  x_tag+=self.alpha*gradients
                  x_tag = x_tag.clamp(min=x-self.eps, max=x+self.eps)
                  x_tag = x_tag.clamp(min=0, max=1)
           
            return  x_tag, q_num_arr

                   

class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        x_tag = x.detach().clone()
        logits = 0.
        if self.rand_init:
            x_tag+=torch.distributions.Uniform(-self.eps, self.eps).sample(x_tag.size()).to(x_tag.device)
        for i in range(self.n):
            for model in self.models:
                  model.zero_grad()
            x_tag.requires_grad = True
            for model in self.models:
                  logits += model(x_tag)
            logits /= len(self.models)
      
           
            if self.early_stop:
                num_correct = (torch.argmax(logits,dim=-1) ==y).sum()
                if ((targeted==True) and (num_correct==y.shape[0])): 
                    break
                if ((targeted==False) and (num_correct == 0)):
                    break
                if targeted:
                    mask = ~mask
            
            loss  = self.loss_func(logits,y).mean()
            loss.backward(retain_graph=True)
            with torch.no_grad():
                  gradients = x_tag.grad.sign()
                  if self.early_stop:
                      gradients*= mask.view(-1, 1, 1, 1)
                  elem = -self.alpha*gradients if targeted else self.alpha*gradients
                  x_tag+=elem
                  x_tag = x_tag.clamp(min=x-self.eps, max=x+self.eps)
                  x_tag = x_tag.clamp(min=0, max=1)
      
     
        return x_tag

