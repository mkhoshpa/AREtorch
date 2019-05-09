import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from attacks import *
from models import Basic_MNIST,load_model,Generator


BASELINE_PATH='pretrained/baseline.pkt'
GAN_PATH = 'pretrained/generator.pkt'
GAN_Y=5
def get_mnist_loader(bs=1,size=(28,28),test=True):

    if(test!= True):

        imageset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transforms.Compose([transforms.ToTensor()]))
        imageloader = torch.utils.data.DataLoader(imageset, batch_size=bs,
                                                shuffle=False, num_workers=2)
    else:
        imageset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transforms.Compose([transforms.ToTensor()]))
        imageloader = torch.utils.data.DataLoader(imageset, batch_size=bs,
                                                shuffle=False, num_workers=2)
                                
    return imageloader

class Evaluate:

    def __init__(self,model,dataset='MNIST',bs=1,device='cpu'):
        self.model=model 
        self.bs=bs
        self.device=device
        if(dataset=='MNIST'):
            self.dataloader=get_mnist_loader(bs=bs)
        else:
            raise NotImplementedError('only MNIST is supported at this time.')

    def restricted_blackbox(self,num=50):
        """
        Performs spsa l-inf, transferability based on PGD(epsilon=0.3) l-inf
        DecisionBoundary L2 attacks

        Parameters
        ----------
        num : 'int'
            number of test sample to be attacked

        """        
        spsa_success =0.0
        original_accuracy=0.0
        tr_success=0.0
        db_success=0.0
        db_dists=[]
        #initiate SPSA
        spsa = SPSA(self.model,0.3,0.05,self.bs,steps=150,device=self.device,lr=0.02)
        #initiate transferability-based attack
        baseline = Basic_MNIST()
        load_model(baseline,BASELINE_PATH)
        transfer_attack = TransferAttack(self.model,baseline,device=self.device)
        #initiate decision-boundary l2 attack
        boundary_attack=BoundryAttack(self.model,iteration=3000,device=self.device)
        #start attacking
        for i,data in enumerate(self.dataloader,0):
            imgs,labels = data
            output=self.model(imgs)
            _,predicted = torch.max(output,1)
            if(predicted == labels):
                original_accuracy += 1
                np_imgs,np_labels = imgs.numpy(),labels.numpy()
                #spsa
                np_adv_x_spsa= spsa.attack(np_imgs,np_labels)
                adv_x_var_spsa = torch.tensor(np_adv_x_spsa,device=self.device)
                output=self.model(adv_x_var_spsa)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    spsa_success += 1
                #transferability
                np_adv_x_tr= transfer_attack.attack(np_imgs,np_labels)
                adv_x_var_tr = torch.tensor(np_adv_x_tr,device=self.device)
                output=self.model(adv_x_var_tr)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    tr_success += 1
                #decision boundary
                np_adv_x_db,dist_db=boundary_attack.attack(np_imgs,np_labels)
                adv_x_var_db = torch.tensor(np_adv_x_db,device=self.device)
                output=self.model(adv_x_var_db)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    db_success += 1
                    db_dists.append(dist_db)
            if(i == num):
                break

        
        spsa_success_percentage =spsa_success / original_accuracy
        tr_success_percentage = tr_success / original_accuracy
        db_success_percentage = db_success /original_accuracy
        db_mean_distance = np.sum(db_dists) / db_success
        original_accuracy_percentage = original_accuracy/num



    def restricted_basics(self,num=50):
        """
        Performs gausian noise  l-inf epsilon=0.3 and salt and pepper l-0 attacks

        Parameters
        ----------
        num : 'int'
            number of test sample to be attacked

        """       
        noise_success =0.0
        original_accuracy=0.0
        saltandpepper_success=0.0
        sp_dists=[]
        #initiate GaussianNoise
        guass_noise = GaussianNoise(self.model)
        #initiate salt and pepper
        sandp = SaltAndPepper(self.model,device=self.device)

        #start attacking
        for i,data in enumerate(self.dataloader,0):
            imgs,labels = data
            imgs,labels=imgs.to(self.device),labels.to(self.device)

            output=self.model(imgs)
            _,predicted = torch.max(output,1)
            if(predicted == labels):
                original_accuracy += 1
                np_imgs,np_labels = imgs.cpu().numpy(),labels.cpu().numpy()
                #gaussian noise
                np_adv_x_g= guass_noise.attack(np_imgs,np_labels)
                adv_x_var_g = torch.tensor(np_adv_x_g,device=self.device)
                output=self.model(adv_x_var_g)
                _,predicted = torch.max(output.data,1)
                if(predicted != labels):
                    noise_success += 1
                #salt and pepper
                np_adv_x_sp, np_sp_dist = sandp.attack(np_imgs,np_labels)
                if(np_adv_x_sp is not None):

                   
                    adv_x_var_sp = torch.tensor(np_adv_x_sp, device=self.device)
                    output=self.model(adv_x_var_sp)
                    _,predicted = torch.max(output,1)
                    if(predicted != labels):
                        saltandpepper_success += 1
                        sp_dists.append(np_sp_dist)
                
            if(i == num):
                break

        
        sp_success_percentage =saltandpepper_success / original_accuracy
        noise_success_percentage = noise_success / original_accuracy
        sp_mean_distance = np.sum(sp_dists) / saltandpepper_success
        original_accuracy_percentage = original_accuracy/num



    def restricted_whitebox(self,num=50):
        """
        Performs CW l2, EAD l1, PGD l-inf with epsilon=0.3 attacks

        Parameters
        ----------
        num : 'int'
            number of test sample to be attacked

        """
        cw_success =0.0
        original_accuracy=0.0
        ead_success=0.0
        cw_dists=[]
        ead_dists=[]
        ead_tar_dists=[]
        pgd_success=0.0
        pgd_tar_success=0.0
        cw_tar_success=0.0
        cw_tar_dists=[]
        #initiate PGD untargeted
        pgd = LinfPGDAttack(self.model,device=self.device)
        #initiate PGD targeted
        pgd_targeted = LinfPGDAttack(self.model,device=self.device,targeted = True)
        #initiate untargeted CW
        cw = AttackCarliniWagnerL2(self.model)
        #initiate targeted cw
        cw_tar = AttackCarliniWagnerL2(self.model,targeted=False)
        #initiate EAD attack
        ead = EAD(self.model, targeted = False)
        #initiate targeted EAD attack
        ead_tar = EAD(self.model,targeted=True)
        #start attacking
        for i,data in enumerate(self.dataloader,0):
            imgs,labels = data
            output=self.model(imgs)
            _,predicted = torch.max(output,1)
            if(predicted == labels):
                original_accuracy += 1
                np_imgs,np_labels = imgs.numpy(),labels.numpy()
                ################### pgd
                np_adv_x_pgd = pgd.attack(np_imgs,np_labels)
                adv_x_var_pgd = torch.tensor(np_adv_x_pgd,device=self.device)
                output=self.model(adv_x_var_pgd)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    pgd_success += 1
                
                ################### pgd targeted
                np_adv_x_pgd_tar = pgd_targeted.attack(np_imgs,np_labels)
                adv_x_var_pgd_tar = torch.tensor(np_adv_x_pgd_tar,device=self.device)
                output=self.model(adv_x_var_pgd_tar)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    pgd_tar_success += 1

                #################### CW
                np_adv_x_cw,l2_dist = cw.attack(np_imgs,np_labels)
                adv_x_var_cw = torch.tensor(np_adv_x_cw,device=self.device)
                output=self.model(adv_x_var_cw)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    cw_success += 1
                    cw_dists.append(l2_dist)
                
                #################### CW targeted
                np_adv_x_cw_tar,l2_dist = cw.attack(np_imgs,np_labels)
                adv_x_var_cw_tar = torch.tensor(np_adv_x_cw_tar,device=self.device)
                output=self.model(adv_x_var_cw_tar)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    cw_tar_success += 1
                    cw_tar_dists.append(l2_dist)
                
                ##################### EAD
                np_adv_x_ead,l1_dist = ead.attack(np_imgs,np_labels)
                adv_x_var_ead = torch.tensor(np_adv_x_ead,device=self.device)
                output=self.model(adv_x_var_ead)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    ead_success += 1
                    ead_dists.append(l1_dist)
                
                ##################### EAD targeted
                np_adv_x_ead_tar,l1_dist = ead_tar.attack(np_imgs,np_labels)
                adv_x_var_ead_tar = torch.tensor(np_adv_x_ead_tar,device=self.device)
                output=self.model(adv_x_var_ead_tar)
                _,predicted = torch.max(output,1)
                if(predicted != labels):
                    ead_tar_success += 1
                    ead_tar_dists.append(l1_dist)
                

        cw_success_percentage =cw_success / original_accuracy
        cw_tar_success_percentage = cw_tar_success / original_accuracy
        ead_success_percentage =ead_success / original_accuracy
        ead_tar_success_percentage = ead_tar_success / original_accuracy
        pgd_success_percentage =pgd_success / original_accuracy
        pgd_tar_success_percentage = pgd_tar_success / original_accuracy


        cw_mean_distance = np.sum(cw_dists) / cw_success
        cw_tar_mean_distance = np.sum(cw_tar_dists) / cw_tar_success
        ead_tar_mean_distance = np.sum(ead_tar_dists) / ead_tar_success
        ead_mean_distance = np.sum(ead_dists) / ead_success

        original_accuracy_percentage = original_accuracy/num



    def unrestricted_whitebox(self,num=50):
        """
        Performs gan attack on the model 

        Parameters
        ----------
        num : 'int'
            number of test sample to be attacked

        """
        gan_success = 0.0
        original_accuracy=0.0
        gan_net = Generator(channels=1)
        load_model(gan_net,GAN_PATH)
        gan_attack = GANAttack(self.model,gan_net)

        for i in range(num):
            z = torch.randn((1, 100, 1, 1), device=self.device,requires_grad=False)
            adv_x_np=gan_attack.attack(z,GAN_Y)
            adv_x_var = torch.tensor(adv_x_np,device=self.device)
            output=self.model(adv_x_var)
            _,predicted = torch.max(output,1)
            if(predicted != GAN_Y):
                gan_success += 1

        gan_success_percentage =gan_success / original_accuracy
        original_accuracy_percentage = original_accuracy/num


 


