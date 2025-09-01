import torch
import torch.nn as nn
import torch.nn.functional as F

def carlini_wagner_loss(outputs, y, large_const=1e6):
    y = F.one_hot(y, outputs.shape[1])
    logits_y = torch.sum(torch.mul(outputs, y), 1)
    logits_max_non_y, _ = torch.max((outputs-large_const* y), 1)
    return logits_max_non_y - logits_y

class PGDAttack:

    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True, loss='ce'):
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        if loss=='ce':
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
        elif loss=='cw':
            self.loss_func = carlini_wagner_loss

    def execute(self, x, y, targeted=False):

        # param to control early stopping
        allow_update = torch.ones_like(y)
        
        # init
        x_adv = torch.clone(x)
        x_adv.requires_grad = True
        if self.rand_init:
            x_adv.data = x_adv.data + self.eps*(2*torch.rand_like(x_adv)-1)
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

        for i in range(self.n):
            # get grad
            outputs = self.model(x_adv)
            loss = torch.mean(self.loss_func(outputs, y))
            loss.backward()
            g = torch.sign(x_adv.grad)

            # early stopping
            if self.early_stop:
                g = torch.mul(g, allow_update[:, None, None, None])

            # pgd step
            if not targeted:
                x_adv.data += self.alpha*g
            else:
                x_adv.data -= self.alpha*g
            x_adv.data = torch.clamp(x_adv, x-self.eps, x+self.eps)
            x_adv.data = torch.clamp(x_adv, 0., 1.)

            # attack success rate
            with torch.no_grad():
                outputs = self.model(x_adv)
                _, preds = torch.max(outputs, 1)
                if not targeted:
                    success = preds!=y
                else:
                    success = (preds==y)
                # early stopping
                allow_update = allow_update - allow_update*success
                if self.early_stop and torch.sum(allow_update)==0:
                    break

        # done
        return x_adv





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