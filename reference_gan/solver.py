import os
import time
import datetime
import torch, gc
import torch.nn as nn
import glob
import os.path as osp
import sys
sys.path.append("..\\reference_gan")

from .model import Generator
from .model import Discriminator
from torchvision.models import vgg19
from torchvision.utils import save_image

from PIL import Image
import numpy as np
from .utils import elastic_transform
from .tps_transformation import tps_transform
from torchvision import transforms as T

import cv2
from matplotlib import pyplot as plt


from scipy import ndimage as ndi
from skimage import io
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.transform import resize

vgg_activation = dict()

def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output.detach()

    return hook



class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.data_loader = data_loader
        self.img_size    = config['MODEL_CONFIG']['IMG_SIZE']
        assert self.img_size in [256]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_recon = config['TRAINING_CONFIG']['LAMBDA_G_RECON']
        self.lambda_g_style = config['TRAINING_CONFIG']['LAMBDA_G_SYTLE']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']
        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp     = config['TRAINING_CONFIG']['LAMBDA_GP']
        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']
        self.mse_loss = nn.MSELoss()

        self.triplet = config['TRAINING_CONFIG']['TRIPLE_LOSS'] == 'True'
        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla']

        if self.triplet:
            self.triplet_loss = nn.TripletMarginLoss(margin=config['TRAINING_CONFIG']['LAMBDA_TR'])
            self.triple_margin = config['TRAINING_CONFIG']['LAMBDA_TR']
            #self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.scaled_dot_product(),
            # margin=config['TRAINING_CONFIG']['LAMBDA_TR'])
            # triplet_loss(anchor, positive, negative)

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss =='vanilla':
            self.adversarial_loss = torch.nn.BCELoss()
        self.l1_loss = torch.nn.L1Loss()

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'
        self.d_spec = config['TRAINING_CONFIG']['D_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # vgg activation
        #self.target_layer = ['relu_3', 'relu_8', 'relu_13', 'relu_17']
        self.target_layer = ['relu_3', 'relu_8']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])
        self.test_dir =  os.path.join(self.train_dir, config['TRAINING_CONFIG']['TEST_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()
        
        self.config = config        

    def build_model(self):
        print("device:" ,self.gpu)
        self.G = Generator(spec_norm=self.g_spec).to(self.gpu)
        self.D = Discriminator(spec_norm=self.d_spec, LR=0.2).to(self.gpu)
        self.vgg = vgg19(pretrained=True)
        for layer in self.target_layer:
            self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))
        self.vgg.to(self.gpu)

        """
        self.vgg.features[3].register_forward_hook(get_activation('relu_3'))
        self.vgg.features[8].register_forward_hook(get_activation('relu_8'))
        self.vgg.features[17].register_forward_hook(get_activation('relu_17'))
        self.vgg.features[26].register_forward_hook(get_activation('relu_26'))
        self.vgg.features[35].register_forward_hook(get_activation('relu_35'))
        """
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        #with open('..\\reference_gan\\' + os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def scaled_dot_product(self, a, b):
        #https://github.com/pytorch/pytorch/issues/18027
        channel = a.size(1)
        scale_factor = torch.sqrt(torch.cuda.FloatTensor([channel]))
        out = torch.bmm(a.view(self.batch_size, 1, channel), b.view(self.batch_size, channel , 1)).reshape(-1)
        #out = torch.div(out, scale_factor)
        return out

    def triple_loss_custom(self, anchor, positive, negative, margin=12):
        distance = self.scaled_dot_product(anchor, positive) - self.scaled_dot_product(anchor, negative) + margin
        #print('distance : ',distance)
        #print('torch.max(distance, torch.zeros_like(distance)) : ', torch.max(distance, torch.zeros_like(distance)))
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return loss

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-G.ckpt'))

        if len(ckpt_list) == 0:
            return 0
        ckpt_list = [int(x.split("\\")[-1].split("-")[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        print("start epoch: ",epoch)
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.G.to(self.gpu)
        self.D.to(self.gpu)
        return epoch

    def image_reporting(self, fixed_sketch, fixed_reference, fixed_elastic_reference, epoch, postfix=''):
        image_report = list()
        image_report.append(fixed_sketch.expand_as(fixed_reference))
        image_report.append(fixed_elastic_reference)
        image_report.append(fixed_reference)
        fake_result, _ = self.G(fixed_elastic_reference, fixed_sketch)
        image_report.append(fake_result)
        x_concat = torch.cat(image_report, dim=3)
        sample_path = os.path.join(self.sample_dir, '{}-images{}.jpg'.format(epoch, postfix))
        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, fixed_elastic_reference, fixed_reference, fixed_sketch = next(data_iter)

        splited_fixed_sketch = list(torch.chunk(fixed_sketch, self.batch_size, dim=0))
        first_fixed_sketch = splited_fixed_sketch[0]
        del splited_fixed_sketch[0]
        splited_fixed_sketch.append(first_fixed_sketch)
        shifted_fixed_sketch = torch.cat(splited_fixed_sketch, dim=0)

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_reference = fixed_reference.to(self.gpu)
        fixed_elastic_reference = fixed_elastic_reference.to(self.gpu)
        shifted_fixed_sketch = shifted_fixed_sketch.to(self.gpu)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    _, elastic_reference, reference, sketch = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, elastic_reference, reference, sketch = next(data_iter)

                elastic_reference = elastic_reference.to(self.gpu)
                reference = reference.to(self.gpu)
                sketch = sketch.to(self.gpu)
                loss_dict = dict()
                if (i + 1) % self.d_critic == 0: 

                    fake_images, _ = self.G(elastic_reference, sketch)
                    d_loss = None

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        real_score = self.D(torch.cat([reference, sketch], dim=1))
                        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))
                        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake
                    elif self.gan_loss == 'wgan':
                        real_score = self.D(torch.cat([reference, sketch], dim=1))
                        fake_score = self.D(torch.cat([fake_images.detach(), sketch], dim=1))
                        d_loss_real = -torch.mean(real_score)
                        d_loss_fake = torch.mean(fake_score)
                        alpha = torch.rand(reference.size(0), 1, 1, 1).to(self.gpu)
                        x_hat = (alpha * reference.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                        out_src = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)
                        d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake + self.lambda_d_gp * d_loss_gp

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss_dict['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss_dict['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()

                    if self.gan_loss == 'wgan':
                        loss_dict['D/loss_pg'] = self.lambda_d_gp * d_loss_gp.item()

                if (i + 1) % self.g_critic == 0:
                    fake_images, q_k_v_list = self.G(elastic_reference, sketch)
                    fake_score = self.D(torch.cat([fake_images, sketch], dim=1))
                    if self.gan_loss in ['lsgan', 'vanilla']:
                        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        g_loss_fake = - torch.mean(fake_score)
                    else:
                        pass
                    g_loss_recon = self.l1_loss(fake_images, reference)

                    fake_activation = dict()
                    real_activation = dict()

                    self.vgg(reference)
                    for layer in self.target_layer:
                        fake_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    self.vgg(fake_images)
                    for layer in self.target_layer:
                        real_activation[layer] = vgg_activation[layer]
                    vgg_activation.clear()

                    g_loss_style = 0
                    g_loss_percep = 0

                    for layer in self.target_layer:
                        g_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])
                        g_loss_style += self.l1_loss(self.gram_matrix(fake_activation[layer]), self.gram_matrix(real_activation[layer]))

                    if self.triplet:
                        anchor = q_k_v_list[0].view(self.batch_size, -1)
                        positive = q_k_v_list[1].contiguous().view(self.batch_size, -1)
                        negative = q_k_v_list[2].contiguous().view(self.batch_size, -1)
                        #g_loss_triple = self.triple_loss_custom(anchor=anchor, positive=positive, negative=negative, margin=self.triple_margin)
                        g_loss_triple = self.triplet_loss(anchor=anchor, positive=positive, negative=negative)

                    g_loss = self.lambda_g_fake * g_loss_fake + \
                    self.lambda_g_recon * g_loss_recon + \
                    self.lambda_g_percep * g_loss_percep + \
                    self.lambda_g_style * g_loss_style
                    if self.triplet:
                        g_loss += 1 * g_loss_triple

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss_dict['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss_dict['G/loss_recon'] = self.lambda_g_recon * g_loss_recon.item()
                    loss_dict['G/loss_style'] = self.lambda_g_style * g_loss_style.item()
                    loss_dict['G/loss_percep'] = self.lambda_g_percep * g_loss_percep.item()
                    if self.triplet:
                        loss_dict['G/loss_triple'] = g_loss_triple.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    self.image_reporting(fixed_sketch, fixed_reference, fixed_elastic_reference, e + 1, postfix='')
                    self.image_reporting(shifted_fixed_sketch, fixed_reference, fixed_elastic_reference, e + 1, postfix='_shifted')
                    print('Saved real and fake images into {}...'.format(self.sample_dir))
            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self):
        self.restore_model()
        #data_loader / getitem
        fid = 0
        test_img_dir = osp.join(self.config['TEST_CONFIG']['IMG_DIR'], self.config['TEST_CONFIG']['MODE'])
        #reference = Image.open(osp.join(test_img_dir, '{}_color.png'.format('00'))).convert('RGB')
        reference = Image.open('./reference.png').convert('RGB')
        #sketch = Image.open(osp.join(test_img_dir, '{}_sketch.png'.format('00'))).convert('L')
        sketch = Image.open( './sketch.png').convert('L')

        if self.config['TRAINING_CONFIG']['DIST'] == 'uniform':
            noise = np.random.uniform(self.config['TRAINING_CONFIG']['A'], self.config['TRAINING_CONFIG']['B'], np.shape(reference))
        else:
            noise = np.random.normal(self.config['TRAINING_CONFIG']['MEAN'], self.config['TRAINING_CONFIG']['STD'], np.shape(reference))

        reference = np.array(reference) + noise
        reference = Image.fromarray(reference.astype('uint8'))

        if self.config['TRAINING_CONFIG']['AUGMENT'] == 'elastic':
            augmented_reference = elastic_transform(np.array(reference), 1000, 8, random_state=None)
            augmented_reference = Image.fromarray(augmented_reference)
        elif self.config['TRAINING_CONFIG']['AUGMENT'] == 'tps':
            augmented_reference = tps_transform(np.array(reference))
            augmented_reference = Image.fromarray(augmented_reference)
        else:
            augmented_reference = reference

        img_transform_gt = list()
        img_transform_sketch = list()
        img_size = self.config['MODEL_CONFIG']['IMG_SIZE']

        img_transform_gt.append(T.Resize((img_size, img_size)))
        img_transform_gt.append(T.ToTensor())
        img_transform_gt.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        img_transform_gt = T.Compose(img_transform_gt)

        img_transform_sketch.append(T.Resize((img_size, img_size)))
        img_transform_sketch.append(T.ToTensor())
        img_transform_sketch.append(T.Normalize(mean=(0.5), std=(0.5)))
        img_transform_sketch = T.Compose(img_transform_sketch)

        fixed_elastic_reference = img_transform_gt(augmented_reference)
        fixed_reference = img_transform_gt(reference)
        fixed_sketch = img_transform_sketch(sketch)

        fixed_sketch = fixed_sketch.to(self.gpu)
        fixed_reference = fixed_reference.to(self.gpu)
        fixed_elastic_reference = fixed_elastic_reference.to(self.gpu)

        print('Start testing...')

        fixed_elastic_reference = torch.stack([fixed_elastic_reference], dim=0)
        fixed_reference = torch.stack([fixed_reference], dim=0)
        fixed_sketch = torch.stack([fixed_sketch], dim=0)

        print(fixed_elastic_reference.size(), fixed_reference.size(), fixed_sketch.size())
          
        with torch.no_grad():
            image_report = list()
            image_report.append(fixed_sketch.expand_as(fixed_reference))
            image_report.append(fixed_elastic_reference)
            image_report.append(fixed_reference)
            fake_result, _ = self.G(fixed_elastic_reference, fixed_sketch)
            image_report.append(fake_result)
            x_concat = torch.cat(image_report, dim=3)
            sample_path = os.path.join(self.test_dir, '{}-images{}.jpg'.format(00, '_test'))
            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
            res_path = os.path.join(self.result_dir, 'gan_image.jpg')
            save_image(self.denorm(fake_result.data.cpu()), res_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(self.sample_dir))
    
        print('Testing is finished')

    def test_seg(self):

        input_path = 'colorization_gan4/tests/1-images_test.jpg'
        image = io.imread(input_path, cv2.IMREAD_COLOR)
        image = resize(image[:,256*3:], (self.img_size, self.img_size))

        sketch_path = 'data/test/2955070_sketch.png'
        sketch = io.imread(sketch_path, cv2.IMREAD_COLOR)
        sketch = resize(sketch, (self.img_size, self.img_size))

        print("image.shape: ", image.shape, type(image))
        print("sketch.shape: ", sketch.shape, type(sketch))

        # denoise image
        denoised = rank.median(image, disk(1)) # 2

        # find continuous region
        markers = rank.gradient(denoised, disk(1)) < 10 # 5, 10
        markers = ndi.label(markers)[0]

        # local gradient
        gradient = rank.gradient(denoised, disk(1)) # 2

        # process the watershed
        labels = watershed(gradient, markers)
        # print("label:", labels.shape)

        #색 지정
        gan_path = 'colorization_gan4/tests/1-images_test.jpg'
        gan_image = cv2.imread(gan_path , cv2.IMREAD_COLOR) # 256, 256, 3
        gan_image = gan_image[:,256*3:]
        cv2.imwrite('colorization_gan4/results/gan_image.jpg',gan_image)

        res_image = gan_image.copy()
    
        colors=[]
        visit = set()
        for i in range(self.img_size):
            for j in range(self.img_size):
                if (i,j) in visit: continue
                queue = list()
                queue.append((i,j))
                visit.add((i,j))
                bgr=[0,0,0]
                cnt=0
                while queue:
                    node = queue.pop(0)
                    x,y = node
                    gbgr = gan_image[x,y]
                    bgr = bgr + gbgr
                    cnt+=1  
                    for dx,dy in [(0,-1),(0,1),(-1,0),(1,0)]: 
                        nx= x+dx
                        ny= y+dy
                        if nx<0 or nx>=self.img_size or ny<0 or ny>=self.img_size or ((nx,ny) in visit) or labels[nx][ny]!=labels[i][j]: continue
                        queue.append((nx,ny))
                        visit.add((nx,ny))
                bgr = bgr // cnt
                colors.append((labels[i][j], bgr))

        #색 넣기 
        #https://bkshin.tistory.com/entry/OpenCV-24-%EC%97%B0%EC%86%8D-%EC%98%81%EC%97%AD-%EB%B6%84%ED%95%A0-%EA%B1%B0%EB%A6%AC-%EB%B3%80%ED%99%98-%EB%A0%88%EC%9D%B4%EB%B8%94%EB%A7%81-%EC%83%89-%EC%B1%84%EC%9A%B0%EA%B8%B0-%EC%9B%8C%ED%84%B0%EC%85%B0%EB%93%9C-%EA%B7%B8%EB%9E%A9%EC%BB%B7-%ED%8F%89%EA%B7%A0-%EC%9D%B4%EB%8F%99-%ED%95%84%ED%84%B0?category=1148027
        for mid, color in colors: # 선택한 라벨 아이디 갯수 만큼 반복
                # 같은 라벨 아이디 값을 갖는 영역을 라벨 선택한 색상으로 채우기 
                res_image[labels==mid] = color

        # display results
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                                sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title("Original")

        ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
        ax[1].set_title("Local Gradient")

        ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
        ax[2].set_title("Markers")

        ax[3].imshow(image, cmap=plt.cm.gray)
        ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
        ax[3].set_title("Segmented")
        
        cv2.imwrite('colorization_gan4/tests/1-images_test_seg_res.jpg', res_image)
        #plt.imsave('colorization_gan4/tests/1-images_test_seg_cmap.jpg', labels, cmap=plt.cm.nipy_spectral)
        plt.imsave('colorization_gan4/tests/1-images_test_seg_cmap.jpg', labels, cmap=plt.cm.viridis)

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()



    

