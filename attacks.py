import copy
import os
import sys
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.nn as nn
from torch import optim
from torch import autograd

from helpers import *





# --- White-box attacks ---

class FGSMAttack:
    def __init__(self, model, epsilon=0.3,device="cuda",targeted=False,clip_min=0, clip_max=1):
        """
        l-inf
        FGSM Attack: https://arxiv.org/abs/1412.6572

        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()
        self.device=device
        self.clip_max=clip_max
        self.clip_min=clip_min
        self.targeted = targeted

    def attack(self, x, y):
        """
        compute x+epsilon*sign(dloss/dx)
        """
    

        adv_x = np.copy(x)

        X_var = torch.tensor(adv_x, requires_grad=True,device=self.device)
        if(self.targeted==False):
            y_var = torch.tensor(y,device=self.device)
            output = self.model(X_var)
            loss = self.loss_fn(output, y_var)

            self.model.zero_grad()
            
            loss.backward()

            with torch.no_grad():
                grad_sign = X_var.grad.data.sign()
                adv_x += self.epsilon * grad_sign

        else:
            target = get_MNIST_CIFAR_target(y)
            target_var = torch.tensor(target,device=self.device)
            output = self.model(X_var)
            loss = self.loss_fn(output, target_var)

            self.model.zero_grad()

            loss.backward()

            with torch.no_grad():
                grad_sign = X_var.grad.data.sign()
                adv_x -= self.epsilon * grad_sign

            if self.clip_min is not None and self.clip_max is not None:
                adv_x = torch.clamp(adv_x, min=self.clip_min, max=self.clip_max)        
            adv_x=adv_x.cpu().numpy()

        return adv_x


class LinfPGDAttack:
    def __init__(self, model, epsilon=0.3, iteration=40, a=0.01, 
        random_start=True,device='cpu',targeted=False,clip_min=0, clip_max=1):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.iteration = iteration
        self.a = a
        self.rand = random_start
        self.loss_fn = nn.CrossEntropyLoss()
        self.device=device
        self.targeted = targeted
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, x, y):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        x_var=torch.tensor(x,device=self.device)
        if self.rand:
            adv_x = x + np.random.uniform(-self.epsilon, self.epsilon,
                x.shape).astype('float32')
        else:
            adv_x = np.copy(x)

        for i in range(self.iteration):
            adv_x_var = torch.tensor(adv_x, requires_grad=True,device=self.device)
            if(self.targeted == False):
                y_var = torch.tensor(y,device=self.device)
                output = self.model(adv_x_var)
                loss = self.loss_fn(output, y_var)

                self.model.zero_grad()

                loss.backward()
                grad_sign = adv_x_var.grad.data.sign()

                with torch.no_grad():
                    adv_x_var += self.a * (grad_sign)
            else:
                target = get_MNIST_CIFAR_target(y)
                target_var = torch.tensor(target,dtype=torch.int32,device=self.device)
                output = self.model(adv_x_var)
                loss = self.loss_fn(output, target_var)

                self.model.zero_grad()

                loss.backward()
                grad_sign = adv_x_var.grad.data.sign()
                with torch.no_grad(): 
                    adv_x_var -= self.a * (grad_sign)
            with torch.no_grad():
              perturb = adv_x_var - x_var
              perturb = torch.clamp(perturb,min=-self.epsilon,max=self.epsilon)
              adv_x_var = x_var+perturb
              adv_x_var = torch.clamp(adv_x_var, self.clip_min, self.clip_max) 
              adv_x=adv_x_var.cpu().numpy()

        return adv_x







"""PyTorch Carlini and Wagner L2 attack algorithm.
partialy adopted from https://github.com/rwightman/pytorch-nips2017-attack-example
Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""



class AttackCarliniWagnerL2:

    def __init__(self,model, targeted=True, search_steps=5, max_steps=2000, device='cuda', debug=False):
        self.model=model
        self.debug = debug
        self.targeted = targeted
        self.num_classes = 10
        self.confidence = 20  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.1  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1.
        self.clip_max = 1.
        self.device = device
        self.clamp_fn = 'tanh'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()

        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        if self.clamp_fn == 'tanh':
            input_adv = tanh_rescale(modifier_var + input_var, self.clip_min, self.clip_max)
        else:
            input_adv = torch.clamp(modifier_var + input_var, self.clip_min, self.clip_max)

        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_np = loss.data[0]
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, dist_np, output_np, input_adv_np

    def attack(self, input,label, batch_idx=0):
        model=self.model
        batch_size = input.size(0)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None
        #if does not provide a target, perform untargeted attack
        if(self.targeted == False):
            target = label
        else:
            target = get_MNIST_CIFAR_target(label) 
        
        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.device=='cuda':
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            modifier = torch.normal(means=modifier, std=0.001)
        if self.device=='cuda':
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(self.binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.device=='cuda':
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist, output, adv_img = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_orig)

                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, dist: {2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                  i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]

                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack,o_best_l2




class TransferAttack:
    def __init__(self,model,base_model,epsilon=0.3,transfer_tries=40, iteration=40, a=0.01, 
        random_start=True,device='cpu',targeted=False):
        self.model=model
        self.base_model=base_model
        self.targeted=targeted
        self.iteration=iteration
        self.device=device
        self.epsilon=epsilon
        self.a=a
        self.random_start=random_start
        self.transfer_tries=transfer_tries


    def attack(self,x,y):
        """
        Takes batch size of 1 samples and searches for adversarial examples.
        returns adversarial examples or None 

        """
        for i in range(self.transfer_tries):
            pgd=LinfPGDAttack(self.base_model,epsilon=self.epsilon,iteration=self.iteration,a=self.a,
            random_start=self.random_start,device=self.device,targeted=self.targeted)
            x_adv=pgd.attack(x,y)
            a_adv_var=torch.tensor(x_adv,device=self.device)
            output=self.base_model(a_adv_var)
            
            if(torch.max(output,1)!=y):
                return x_adv
            
        return


class GANAttack:
    """
    Generate adversarial examples using class conditional generative adversarial networks
    """
    def __init__(self,model,generator,iteration=100,g_output_mul=0.5,g_output_add=0.5,device='cpu',lr=0.01,targeted=False):
        self.model=model
        self.generator=generator
        self.iteration=iteration
        self.device=device
        self.g_output_mul=g_output_mul
        self.g_output_add=g_output_add
        self.loss_fn = nn.CrossEntropyLoss()
        self.targeted=targeted
        
    def attack(self,initial_z,y):
        """
        Takes initial noise and true label of class conditional generator and produces adversarial examples
        """
        if(self.targeted == True):
            target = get_MNIST_CIFAR_target(y)
        initial_z_var=torch.tensor(initial_z,requires_grad=True,device=self.device)
        optimizer = optim.Adam([initial_z_var], lr=0.03)
        adv_x_var=self.generator(initial_z_var)

        for i in range(self.iteration):
            adv_x_var=self.generator(initial_z_var)
            adv_x_var=adv_x_var.mul(self.g_output_mul).add(self.g_output_add)
            output=self.model(adv_x_var)
            _, predicted = torch.max(output.data, 1)
            if(self.targeted == False):
                loss=(-1) * self.loss_fn(output,y)
            else:
                loss= self.loss_fn(output,target)
            self.model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        adv_x= adv_x_var.cpu().numpy()
        return adv_x



        
class SPSA:
    """
    a gradient free optimization attack.
    we use original psudo code fro  the paper.
    using adam optimizer:https://arxiv.org/pdf/1412.6980.pdf
    link to the paper:https://arxiv.org/pdf/1802.05666.pdf
    """
    def __init__(self,model,epsilon,delta,batch_size,targeted=False,steps=100,device='cpu',lr=0.01):
        self.model=model
        self.epsilon=epsilon
        self.delta=delta
        self.steps=steps
        self.batch_size=batch_size
        self.device=device
        self.lr=lr
        self.loss=nn.CrossEntropyLoss()
        self.targeted=targeted

    def attack(self,x,y,target=None):
        if(target is None):
            self.targeted=False
        y_var=torch.tensor(y,device=self.device)
        x_var=torch.tensor(x,device=self.device,dtype=torch.float)
        #current accuracy
        output=self.model(x_var)
        _,predicted=torch.max(output,1)

        #random start
        batch_v=np.ones_like(x)
        noise=torch.tensor(batch_v,device=self.device).uniform_(-2, 2).sign()
        noise=noise*self.epsilon
        x_t=x_var+noise
        x_t = torch.clamp(x_t, 0, 1)

        
        m=0
        v=0
        beta1=0.9
        beta2=0.999
        for i in range(self.steps):
            print(i)
            g_var=self.gradient(x_t.detach().cpu().numpy(),y,target)
            g=(1/self.batch_size)*torch.sum(g_var,0)
            #m=beta1*m+(1-beta1)*g
            #v=beta2*v+(1-beta2)*g.pow(2)
            #m_hat=m/(1-beta1 ** (i+1))
            #v_hat=v/(1-beta2 ** (i+1))
            #x_t=x_t-self.lr* m_hat/(v_hat ** 0.5 + 0.00000001)
            
            x_t=x_t-self.lr*g.sign()
            diff= x_t-x_var
            diff=torch.clamp(diff,(-1)* self.epsilon,self.epsilon)
            x_t = x_var+diff
            x_t = torch.clamp(x_t, 0, 1)
            

        adv_x=x_t.cpu().numpy()
        return adv_x

    def loss_fn(self,x,y,target):

        output=self.model(x)
        if(self.targeted== False):
            loss=(-1)*self.loss(output,y)
        else:
            loss=self.loss(output,target)
        return loss
      
    def gradient(self,x,y,target):
      x_var=torch.tensor(x,device=self.device,dtype=torch.float)
      y_var=torch.tensor(y,device=self.device)
      x_t=torch.tensor(x,device=self.device,dtype=torch.float)
      
      batch_v=np.ones_like(x)
      batch_v_var=torch.tensor(batch_v,device=self.device).uniform_(-2, 2)
      batch_v_var=batch_v_var.sign()
      g=[]
      for v_var in batch_v_var:
        g.append(((self.loss_fn(x_t+self.delta*v_var,y_var,target)-self.loss_fn(x_t-self.delta*v_var,y_var,target))/(2*self.delta*v_var)).detach().cpu().numpy())
      g_np=np.array(g)
      g_np=g_np.astype(np.float)
      g_var=torch.tensor(g_np,device=self.device,dtype=torch.float)
      return g_var

"""
EAD attack: https://arxiv.org/abs/1709.04114
Partialy adopted from https://github.com/rwightman/pytorch-nips2017-attack-example
"""
class EAD:
    def __init__(self,model, targeted=True, search_steps=5, max_steps=1000, device='cuda', debug=False,lr=0.001,beta=0.01):
        self.model=model
        self.debug = debug
        self.targeted = targeted
        self.num_classes = 10
        self.confidence = 20  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = 0.1  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 5
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or 1000
        self.abort_early = True
        self.clip_min = -1.
        self.clip_max = 1.
        self.device = device
        self.clamp_fn = 'None'  # set to something else perform a simple clamp instead of tanh
        self.init_rand = False  # an experiment, does a random starting point help?
        self.lr=0.0005
        self.beta=beta
        self.targeted = targeted
    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            if self.targeted:
                output[target] -= self.confidence
            else:
                output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)

        loss2 = dist.sum()

        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var_x,modifier_var_y, target_var, scale_const_var,zt, input_orig=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max
        input_adv=modifier_var_y.clone().requires_grad_()
        output = model(input_adv)

        # distance to the original input data
        if input_orig is None:
            l2dist = l2_dist(input_adv, input_var, keepdim=False)
        else:
            l2dist = l2_dist(input_adv, input_orig, keepdim=False)

        loss = self._loss(output, target_var, l2dist, scale_const_var)

        optimizer.zero_grad()
        loss.backward()
        input_adv_grad=input_adv.grad.data
        with torch.no_grad:
            new_modifier_var_x=self.epst(modifier_var_y-self.lr*input_adv_grad,input_var)
            modifier_var_y=new_modifier_var_x+zt*(new_modifier_var_x-modifier_var_x)
            modifier_var_x=new_modifier_var_x

        l1dist= (modifier_var_x-input_var).sum()
        loss = loss+self.beta*l1dist
        loss_np = loss.data[0]
        l2dist_np = l2dist.data.cpu().numpy()
        l1dist_np = l1dist.data.cpu().numpy()

        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.permute(0, 2, 3, 1).cpu().numpy()  # back to BHWC for numpy consumption
        return loss_np, l2dist_np,l1dist_np, output_np, input_adv_np



    def epst(self,x,x_0):
        cond1 = [(x-x_0)>self.beta]
        cond2 = [torch.abs(x-x_0)<self.beta]
        cond3 =  [(x-x_0)< -1*self.beta]

        upper=torch.min(x-self.beta,torch.zeros_like(x)+1)
        lower=torch.max(x+self.beta,torch.zeros_like(x))

        out= cond1*upper+cond3*lower+cond2*x
        return out

    def attack(self, input,label, batch_idx=0):
        model=self.model
        batch_size = input.size(0)
        global_step = torch.tensor(0,requires_grad=False)
        global_step_t = global_step.clone().astype(torch.float32)
        zt=global_step_t/(global_step_t+3)
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_l1 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_attack = input.permute(0, 2, 3, 1).cpu().numpy()

        # setup input (image) variable, clamp/scale as necessary
        if self.clamp_fn == 'tanh':
            # convert to tanh-space, input already int -1 to 1 range, does it make sense to do
            # this as per the reference implementation or can we skip the arctanh?
            input_var = autograd.Variable(torch_arctanh(input), requires_grad=False)
            input_orig = tanh_rescale(input_var, self.clip_min, self.clip_max)
        else:
            input_var = autograd.Variable(input, requires_grad=False)
            input_orig = None
        if(self.targeted == False):
            target = label
        else:
            target = get_MNIST_CIFAR_target(label) 
        
        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.device=='cuda':
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = input_var.clone()
        if self.init_rand:
            # Experiment with a non-zero starting point...
            noise = torch.normal(means=modifier, std=0.001)
            modifier = input_var.clone()+noise
        if self.device=='cuda':
            modifier = modifier.cuda()
        modifier_var = autograd.Variable(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=0.0005)

        for search_step in range(self.binary_search_steps):
            print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
            if self.debug:
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_l1 = [1e10] * batch_size
            global_step = torch.tensor(0,requires_grad=False)
            global_step_t = global_step.clone().astype(torch.float32)
            zt=global_step_t/(global_step_t+3)
            best_score = [-1] * batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.device=='cuda':
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = autograd.Variable(scale_const_tensor, requires_grad=False)

            prev_loss = 1e6
            for step in range(self.max_steps):
                # perform the attack
                loss, dist,distl1, output, adv_img = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    zt,
                    input_orig)

                if step % 100 == 0 or step == self.max_steps - 1:
                    print('Step: {0:>4}, loss: {1:6.4f}, distl2: {2:8.5f}, distl1:{2:8.5f}, modifier mean: {3:.5e}'.format(
                        step, loss, dist.mean(),distl1.mean(), modifier_var.data.mean()))

                if self.abort_early and step % (self.max_steps // 10) == 0:
                    if loss > prev_loss * .9999:
                        print('Aborting early...')
                        break
                    prev_loss = loss

                # update best result found
                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = distl1[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} distl1: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l1[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist1: {1:.5f}, new dist1: {2:.5f}'.format(
                                  i, best_l1[i], di))
                        best_l1[i] = di
                        best_score[i] = output_label
                    if di < o_best_l1[i] and self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist1: {1:.5f}, new dist1: {2:.5f}'.format(
                                  i, o_best_l1[i], di))
                        o_best_l1[i] = di
                        o_best_score[i] = output_label
                        o_best_attack[i] = adv_img[i]
                

                sys.stdout.flush()
                # end inner step loop

            global_step_t = global_step_t
            zt=global_step_t/(global_step_t+3)

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(best_score[i], target[i]) and best_score[i] != -1:
                    # successful, do binary search and divide const by two
                    upper_bound[i] = min(upper_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if self.debug:
                        print('{0:>2} successful attack, lowering const to {1:.3f}'.format(
                            i, scale_const[i]))
                else:
                    # failure, multiply by 10 if no solution found
                    # or do binary search with the known upper bound
                    lower_bound[i] = max(lower_bound[i], scale_const[i])
                    if upper_bound[i] < 1e9:
                        scale_const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        scale_const[i] *= 10
                    if self.debug:
                        print('{0:>2} failed attack, raising const to {1:.3f}'.format(
                            i, scale_const[i]))
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                else:
                    batch_failure += 1

            print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack,o_best_l1







class GaussianNoise:
    """
    A weak attack that just picks a random point in the attacker's action space.
    When combined with an attack bundling function, this can be used to implement
    random search.
    References:
    https://arxiv.org/abs/1802.00420 recommends random search to help identify
        gradient masking
    https://openreview.net/forum?id=H1g0piA9tQ recommends using noise as part
        of an attack building recipe combining many different optimizers to yield
        a strong optimizer.
    Args:
        model: Model
        dtype: dtype of the data
        kwargs: passed through the super constructor
    """

    def __init__(self, model, dtype=None, eps=0.3, ord=np.inf, clip_min=0, clip_max=1,device=device):


        self.device=device
        self.model=model
        self.epsilon=eps
        self.ord=ord
        self.clip_min=clip_min
        self.clip_max=clip_max
        if(dtype is None):
            self.dtype=list(self.model.parameters())[0].dtype
        else: 
            self.dtype=dtype            
        self.std = self.epsilon / (3**0.5) * (clip_max - clip_min)
        
    def attack(self, x,y=None):

        """
        Generates adversarial weak adversaril examples for a batch x by adding guassian noise
        Args:
            x:inputs.
        """

        if self.ord != np.inf:
            raise NotImplementedError(self.ord)
        x_var = torch.tensor(x,device=self.device)
        noise = torch.zeros_like(x_var,device=self.device,dtype=self.dtype).normal_()*self.std
        adv_x = x_var + noise
        adv_x = torch.clamp(adv_x, min=self.clip_min, max=self.clip_max)

        return adv_x


class SaltAndPepper:
    """
    Line search to find the right amont of salt and pepper noise. 
    Based on foolbox implementation https://arxiv.org/pdf/1707.04131.pdf
    """
    def __init__(self,model,search_steps=10,clip_max=1,clip_min=0,dtype=None,device='cpu'):

        self.model=model
        self.search_steps=search_steps
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.device=device
        if(dtype==None):
            self.dtype=list(self.model.parameters())[0].dtype
        else: 
            self.dtype=dtype

    def attack(self,x,y,targeted=None):

        """
        x: a numpy array containing a batch of one input
        y: trye label of x, int
        targeted: bool, 
        dtype: dtype

        returns: adversarial example: tensor, batch of one and max probibility for salt and pepper attack
        """
        if(targeted is not None):
            raise NotImplementedError("targeted SaltAndPepper attack not implemented")

        x_var=torch.tensor(x,dtype=self.dtype,device=self.device)
        salt=x_var.clone()
        pepper=x_var.clone()
        r=self.clip_max-self.clip_min
        for p in torch.linspace(0,1,self.search_steps):
            p=p.to(self.device)
            salt=x_var.clone().uniform_(0, 1)
            pepper=x_var.clone().uniform_(0, 1)
            salt=(salt>= 1-p/2)*r
            salt=torch.tensor(salt,dtype=self.dtype,device=self.device)
            pepper=-1*(pepper<p)*r
            pepper=torch.tensor(pepper,dtype=self.dtype,device=self.device)


            adv_x=torch.clamp(x_var+salt+pepper,min=self.clip_min,max=self.clip_max)
            _,predict=torch.max(self.model(adv_x),1)
            if(predict.sum() != y[0]):
                max_epsilon = torch.min(torch.tensor(1,device=self.device,dtype=self.dtype), p * 1.2)
                return adv_x,max_epsilon
        return None,None


class BoundryAttack:

    def __init__(self,model,iteration=2000,delta=0.1,epsilon=0.005,clip_max=1,clip_min=0,dtype=None,device='cpu'):
        self.model=model
        self.iteration=iteration
        self.clip_max=clip_max
        self.clip_min=clip_min
        self.device=device
        self.delta=delta
        self.epsilon=epsilon
        if(dtype==None):
            self.dtype=list(self.model.parameters())[0].dtype
        else: self.dtype=dtype
    def attack(self,x,y,targeted=None):
        x_var=torch.tensor(x,device=self.device)
        if targeted != None:
            raise NotImplementedError('targeted decision boundry attack')
        while(True):
            adv_x_var = torch.zeros_like(x_var,device=self.device).uniform_(self.clip_min,self.clip_max)
            _,predicted=torch.max(self.model(adv_x_var),1)
           
            if(predicted != y[0]):
              print(self.l2_dist(adv_x_var-x_var))
              perturb = adv_x_var-x_var
              epsilon=0.1
              while(True):
                new_perturb=(perturb/self.l2_dist(perturb))*(epsilon)
                new_adv = x_var+new_perturb
                _,predicted=torch.max(self.model(new_adv),1)
                if(predicted != y[0]):
                  adv_x_var=new_adv
                  print('found')
                  print(self.l2_dist(adv_x_var-x_var))

                  break
                else:
                  epsilon=epsilon/0.9
              break
            
                
        delta=self.delta
        for i in range(self.iteration):
          delta=self.delta

          orth_step=0
          
          while(True):
              e_step=0
              
              orth_step+=1
              if(orth_step>100):
                break
              p=0
              for k in range(10):
                  noise = torch.zeros_like(x_var,device=self.device).uniform_(self.clip_min,self.clip_max) - adv_x_var
                  noise = noise * self.l2_dist(noise)
                  noise *= delta*self.l2_dist(adv_x_var,x_var)
                  perturbated = adv_x_var+noise
                  #perturbated=torch.clamp(perturbated,self.clip_min,self.clip_max)
                  noise = perturbated - x_var
                  dist_var= adv_x_var-x_var
                  #print("dist_var:")
                  #print(self.l2_dist(dist_var))
                  new_dist_var = perturbated-x_var

                  new_dist_var_on_sphere_i = (new_dist_var/self.l2_dist(new_dist_var))*self.l2_dist(dist_var)
                  _,predicted=torch.max(self.model(torch.clamp(new_dist_var_on_sphere_i+x_var,self.clip_min,self.clip_max)),1)
                  if(predicted != y[0]):
                    p += 1
                    new_dist_var_on_sphere =new_dist_var_on_sphere_i.clone()
              
              

              if(p>0):
                  print(p)
                  if(p>7):
                      delta /=0.9
                  if(p<3):
                    delta *=0.9
                 
                  #take epsilon step
                  epsilon=0.9
                  
                  while(True):
                    e_step+=1
                    if(e_step>200):
                      break
                    #print("new_dist_var_on_sphere:")
                    #print(self.l2_dist(new_dist_var_on_sphere))
                    #print(self.l2_dist(dist_var))
                    ##take a movement tworads original input
                    new_dist_var_after_step = (new_dist_var_on_sphere/self.l2_dist(new_dist_var_on_sphere))*((1.0-epsilon)*self.l2_dist(dist_var))
                    #print("new_dist_var_after_step:")
                    #print(self.l2_dist(new_dist_var_after_step))
                    #imshow(new_dist_var_after_step[0][0].cpu().numpy())
                    potential_adv_x_var = x_var+ new_dist_var_after_step
                    potential_adv_x_var=torch.clamp(potential_adv_x_var,self.clip_min,self.clip_max)
                    _,predicted=torch.max(self.model(potential_adv_x_var),1)
                    #imshow(potential_adv_x_var[0][0].cpu().numpy())
                    #print(self.l2_dist(potential_adv_x_var-x_var))
                    #print(predicted)

                    if(predicted != y[0]):
                        print(epsilon)
                        print(delta)
                        print(predicted)
                        print('here')
                        print(self.l2_dist(potential_adv_x_var,x_var))

                        adv_x_var = potential_adv_x_var
                        break
                    else:
                      epsilon=epsilon*0.5
                  break
              else:
                  delta=delta*0.9

            
        #adv_x_var = potential_adv_x_var
            
        return adv_x_var.cpu().numpy(),self.l2_dist(adv_x_var,x_var).cpu().numpy()



            
    def l2_dist(self,x,y=None):
        if(y is None):
            return x.pow(2).sum().pow(0.5)
        else:
            return (x-y).pow(2).sum().pow(0.5)
