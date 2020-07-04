''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, zs_, ys_, ema, state_dict, time, config):
    def train(x, y):
        G.optim.zero_grad()
        D.optim.zero_grad()
        inner_iter_count = 0
        partial_test_input = 0
        # How many chunks to split x and y into?
        #x = torch.split(x, config['batch_size'])
        #y = torch.split(y, config['batch_size'])
        #print('x len{}'.format(len(x)))
        #print('y len{}'.format(len(y)))
        #assert len(x) == config['num_D_accumulations'] == len(y)
        #D_fake, D_real, G_fake, gy = None, None, None, None
        # Optionally toggle D and G's "require_grad"
        if config['toggle_grads']:
            utils.toggle_grad(D, True)
            utils.toggle_grad(G, False)

        for step_index in range(config['num_D_steps']):
            # If accumulating gradients, loop multiple times before an optimizer step
            D.optim.zero_grad()

            d_reals = None#[None for _ in x]
            g_fakes = None#[None for _ in x]
            #gys = [None for _ in x]
            #zs = [None for _ in x]
            #zs_.sample_()
            #ys_.sample_()
            #gy = ys_[:config['batch_size']]
            #z = zs_[:config['batch_size']].view(zs_.size(0), 9, 8, 8)[:, :5]
            if state_dict['epoch'] < 0:
                #for accumulation_index in range(config['num_D_accumulations']):  # doesn't mean anything right now
                # for fb_iter in range(config['num_feedback_iter']):
                # if fb_iter == 0:
                # z_ = zs_[:config['batch_size']]
                # gy = ys_[:config['batch_size']]
                # print('z_ shape {}'.format(z_.shape))
                # z_ = z_.view(zs_.size(0), 9, 8, 8)[:, :5]
                zs_.sample_()
                z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:,20]  # [:, :5]
                #z_ = z_.view(z_.size(0), -1)

                # zs[accumulation_index] = z
                # z_ = torch.cat([z, torch.zeros(zs_.size(0), 4, 8, 8).cuda()], 1)

                ys_.sample_()
                gy = ys_[:config['batch_size']]
                # gys[accumulation_index] = gy.detach()
                # else:
                # D_real = D_real#.repeat(1,3,1,1)# * g_fakes[accumulation_index]
                # print('zs_ shape 0 {}'.format(zs_.shape))
                # print('\n\n\n\n')
                # print('r shape {}'.format(r.shape))
                # print('g fake shape {}'.format(g_fakes[accumulation_index].shape))
                # print('\n\n\n\n')
                # z_ = zs_[:config['batch_size']].view(zs_.size(0), 9, 8, 8)[:, :8]
                # G_fake = nn.AvgPool2d(4)(g_fakes[accumulation_index])
                # print('z shape 5 {}'.format(z_.shape))
                # z_=z_[:,:3]
                # print('z shape 10 {}'.format(z_.shape))

                # z_ = torch.cat([d_reals[accumulation_index], G_fake, zs[accumulation_index]], 1)
                # print('z shape 15 {}'.format(z_.shape))
                # gy = gys[accumulation_index]
                D_fake, D_real, G_fake = GD(z_,
                                            gy,
                                            x=x,#[accumulation_index],
                                            dy=y,#[accumulation_index],
                                            train_G=False,
                                            split_D=config['split_D'])
                #print('D shape {}'.format(D_fake.shape))
                #print('G fake shape {}'.format(nn.AvgPool2d(4)(G_fake).shape))
                #print('D real shape {}'.format(D_real.shape))
                #print('z shape {}'.format(z_.shape))

                if state_dict['itr'] % 1000 == 0: ##and accumulation_index == 6:
                    print('saving img')
                    torchvision.utils.save_image(x.float().cpu(),#[accumulation_index].float().cpu(),
                                                 '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_pre_xreal.jpg'.format(
                                                     time, state_dict['itr']),
                                                 nrow=int(D_fake.shape[0] ** 0.5), normalize=True)
                    torchvision.utils.save_image(D_fake.float().cpu(),
                                                 '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_pre_dfake.jpg'.format(
                                                     time, state_dict['itr']),
                                                 nrow=int(D_fake.shape[0] ** 0.5), normalize=True)
                    torchvision.utils.save_image(D_real.float().cpu(),
                                                 '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_pre_dreal.jpg'.format(
                                                     time, state_dict['itr']),
                                                 nrow=int(D_fake.shape[0] ** 0.5), normalize=True)

                # d_reals[accumulation_index] = D_real.detach()
                # g_fakes[accumulation_index] = G_fake.detach()

                # Compute components of D's loss, average them, and divide by
                # the number of gradient accumulations
                D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                D_loss = (D_loss_real + D_loss_fake)# / float(config['num_D_accumulations'])
                D_loss.backward()
                # counter += 1

                    # Optionally apply ortho reg in D
                if config['D_ortho'] > 0.0:
                    # Debug print to indicate we're using ortho reg in D.
                    print('using modified ortho reg in D')
                    utils.ortho(D, config['D_ortho'])

                D.optim.step()
                # D.optim.zero_grad()
                # Optionally toggle "requires_grad"
            else:
                for fb_iter in range(config['num_feedback_iter']):
                    #for accumulation_index in range(config['num_D_accumulations']): #doesn't mean anything right now
                    #for fb_iter in range(config['num_feedback_iter']):
                    zs_.sample_()
                    z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:, :20]
                    ys_.sample_()
                    gy = ys_[:config['batch_size']]

                    if fb_iter <= 1:
                        # z_ = zs_[:config['batch_size']]
                        # gy = ys_[:config['batch_size']]
                        #print('z_ shape {}'.format(z_.shape))
                        #z_ = z_.view(zs_.size(0), 9, 8, 8)[:, :5]

                        #zs_.sample_()
                        #z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:, :20]
                        #zs[accumulation_index] = z_
                        #print('three channel x input train D shape before {}'.format(x[:, :3].shape))
                        init_x = nn.AvgPool2d(4)(x[:, :3])
                        z_ = torch.cat([z_, init_x, torch.ones(zs_.size(0), 1, 8, 8).cuda()], 1)
                        #print('three channel x input train D shape after {}'.format(nn.AvgPool2d(4)(x[:, :3]).shape))

                        #ys_.sample_()
                        #gy = ys_[:config['batch_size']]
                        #gys[accumulation_index] = gy.detach()
                    else:
                        #D_real = D_real#.repeat(1,3,1,1)# * g_fakes[accumulation_index]
                        #print('zs_ shape 0 {}'.format(zs_.shape))
                        #print('\n\n\n\n')
                        #print('r shape {}'.format(r.shape))
                        #print('g fake shape {}'.format(g_fakes[accumulation_index].shape))
                        #print('\n\n\n\n')
                        #z_ = zs_[:config['batch_size']].view(zs_.size(0), 9, 8, 8)[:, :8]
                        G_fake = 0.1 * g_fakes + 0.9 * init_x#[accumulation_index]
                        #print('z shape 5 {}'.format(z_.shape))
                        #z_=z_[:,:3]
                        #print('z shape 10 {}'.format(z_.shape))

                        #z_ = torch.cat([zs[accumulation_index],d_reals[accumulation_index], G_fake,], 1)
                        z_ = torch.cat([z_, G_fake, d_reals#[accumulation_index]
                                           ,], 1)
                    #z_ = z_.view(z_.size(0),-1)
                        #print('z shape 15 {}'.format(z_.shape))
                        #gy = gys[accumulation_index]
                    # if state_dict['itr'] % 42 == 0:
                    #     partial_test_input = partial_test_input + torch.cat([g_fakes, d_fakes])
                    D_fake, D_real, G_fake = GD(z_,
                                        gy,
                                        x=x,#[accumulation_index],
                                        dy=y,#[accumulation_index],
                                        train_G=False,
                                        split_D=config['split_D'])
                    #print('D shape {}'.format(D_fake.shape))
                    if state_dict['itr'] % 1000 == 0:# and accumulation_index == 6:
                        print('saving img')
                        torchvision.utils.save_image(x.float().cpu(),#[accumulation_index].float().cpu(),
                                                     '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_xreal.jpg'.format(
                                                         time, state_dict['itr'], fb_iter),
                                                     nrow=int(D_fake.shape[0] ** 0.5), normalize=True)
                        torchvision.utils.save_image(D_fake.float().cpu(),
                        '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_dfake.jpg'.format(
                            time,state_dict['itr'],fb_iter),nrow=int(D_fake.shape[0] ** 0.5),normalize=True)
                        torchvision.utils.save_image(D_real.float().cpu(),
                        '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_dreal.jpg'.format(
                            time,state_dict['itr'],fb_iter),nrow=int(D_fake.shape[0] ** 0.5),normalize=True)


                    D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                    if not fb_iter == 0:
                        d_real_enforcement = losses.loss_enforcing(d_reals#[accumulation_index]
                                                                   , D_real)
                        g_fakes_enforcement = losses.loss_enforcing(g_fakes#[accumulation_index]
                                                                    , nn.AvgPool2d(4)(G_fake))
                        D_loss = (D_loss_real + D_loss_fake + d_real_enforcement + g_fakes_enforcement)# / float(config['num_D_accumulations'])
                    else:
                        D_loss = (D_loss_real + D_loss_fake)# / float(config['num_D_accumulations'])

                    #d_reals[accumulation_index] = D_real.detach()
                    d_reals = D_real.detach()

                    #g_fakes[accumulation_index] = nn.AvgPool2d(4)(G_fake).detach()
                    g_fakes = nn.AvgPool2d(4)(G_fake).detach()

                    # Compute components of D's loss, average them, and divide by
                    # the number of gradient accumulations

                    # D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
                    # if not fb_iter == 0:
                    #     D_loss = (D_loss_real + D_loss_fake + d_real_enforcement + g_fakes_enforcement) / float(config['num_D_accumulations'])
                    # else:
                    #     D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

                    D_loss.backward()

                    #counter += 1

                    # Optionally apply ortho reg in D
                    if config['D_ortho'] > 0.0:
                        # Debug print to indicate we're using ortho reg in D.
                        print('using modified ortho reg in D')
                        utils.ortho(D, config['D_ortho'])

                    D.optim.step()
                        #D.optim.zero_grad()

            # Optionally toggle "requires_grad"

        if config['toggle_grads']:
            utils.toggle_grad(D, False)
            utils.toggle_grad(G, True)

        # Zero G's gradients by default before training G, for safety
        G.optim.zero_grad()

        #d_fakes = [None for _ in range(config['num_G_accumulations'])]
        #g_fakes = [None for _ in range(config['num_G_accumulations'])]
        #gys = [None for _ in range(config['num_G_accumulations'])]
        #for fb_iter in range(config['num_feedback_iter']):
        # If accumulating gradients, loop multiple times
        d_fakes = None#[None for _ in x]
        g_fakes = None#[None for _ in x]
        #gys = [None for _ in x]
        #zs = [None for _ in x]
        if state_dict['epoch'] < 0:
            #for accumulation_index in range(config['num_G_accumulations']):  # doesn't mean anything right now
            zs_.sample_()
            z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:, :20]
            #zs[accumulation_index] = z_[:, :5]
            # z_ = torch.cat([z, torch.zeros(zs_.size(0), 4, 8, 8).cuda()],1)
            ys_.sample_()
            gy = ys_
            #gys[accumulation_index] = gy.detach()

            # D_fake = D_fake.repeat(1,3,1,1)
            # z_ = zs_[:config['batch_size']].view(zs_.size(0), 9, 8, 8)[:, :5]
            #G_fake = nn.AvgPool2d(4)(g_fakes[accumulation_index])
            #z_ = torch.cat([d_fakes[accumulation_index], G_fake, zs[accumulation_index]], 1)
             #   gy = gys[accumulation_index]
            z_ = z_.view(z_.size(0), -1)
            D_fake, G_z = GD(z=z_, gy=gy, train_G=True, split_D=config['split_D'], return_G_z=True)
            G_loss = losses.generator_loss(D_fake)# / float(config['num_G_accumulations'])
            G_loss.backward()

            if state_dict['itr'] % 1000 == 0:# and accumulation_index == 6:
                print('saving img')
                torchvision.utils.save_image(D_fake.float().cpu(),
                                             '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_pre_dfake.jpg'.format(
                                                 time,
                                                 state_dict['itr'],),
                                             nrow=int(D_fake.shape[0] ** 0.5),
                                             normalize=True)
                torchvision.utils.save_image(G_z.float().cpu(),
                                             '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_pre_G_z.jpg'.format(
                                                 time,
                                                 state_dict['itr'],),
                                             nrow=int(D_fake.shape[0] ** 0.5),
                                             normalize=True)

            #g_fakes[accumulation_index] = G_z.detach()
            #d_fakes[accumulation_index] = D_fake.detach()
            # Optionally apply modified ortho reg in G
            if config['G_ortho'] > 0.0:
                print('using modified ortho reg in G')  # Debug print to indicate we're using ortho reg in G
                # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
                utils.ortho(G, config['G_ortho'],
                            blacklist=[param for param in G.shared.parameters()])
            G.optim.step()
            # G.optim.zero_grad()
        else:
            for fb_iter in range(config['num_feedback_iter']):
                #for accumulation_index in range(config['num_G_accumulations']): #doesn't mean anything right now
                zs_.sample_()
                z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:, :20]
                ys_.sample_()
                gy = ys_

                if fb_iter <= 1:
                    #zs_.sample_()
                    #z_ = zs_[:config['batch_size']].view(zs_.size(0), 24, 8, 8)[:, :20]

                    #zs[accumulation_index] = z_
                    #print('three channel x input train G shape before {}'.format(x.shape))
                    init_x = nn.AvgPool2d(4)(x[:, :3])
                    z_ = torch.cat([z_, init_x, torch.ones(zs_.size(0), 1, 8, 8).cuda()], 1)
                    #print('three channel x input train G shape after {}'.format(nn.AvgPool2d(4)(x[:, :3]).shape))
                    #ys_.sample_()
                    #gy = ys_
                    #gys[accumulation_index] = gy.detach()
                else:
                    #D_fake = D_fake.repeat(1,3,1,1)
                    #z_ = zs_[:config['batch_size']].view(zs_.size(0), 9, 8, 8)[:, :5]
                    #G_fake = g_fakes#[accumulation_index]
                    G_fake = 0.1 * g_fakes + 0.9 * init_x  # [accumulation_index]

                    #z_ = torch.cat([zs[accumulation_index], d_fakes[accumulation_index], G_fake, ], 1)
                    z_ = torch.cat([z_, G_fake, d_fakes #[accumulation_index]
                                       ,], 1)
                    if ((not (state_dict['itr'] % config['save_every'])) or (not (state_dict['itr'] % config['test_every']))):
                        partial_test_input = partial_test_input + torch.cat([G_fake, d_fakes], 1)
                        inner_iter_count = inner_iter_count + 1
                    #gy = gys[accumulation_index]
                #z_ = z_.view(z_.size(0), -1)
                D_fake, G_z = GD(z=z_, gy=gy, train_G=True, split_D=config['split_D'], return_G_z=True)

                if not fb_iter == 0:
                    g_fakes_enforcement = losses.loss_enforcing(g_fakes#[accumulation_index]
                                                                , nn.AvgPool2d(4)(G_z))
                    d_fakes_enforcement = losses.loss_enforcing(d_fakes#[accumulation_index]
                                                                , D_fake)
                    G_loss = (losses.generator_loss(D_fake) + 0.1 * g_fakes_enforcement + 0.1 * d_fakes_enforcement) #/ float(config['num_G_accumulations'])
                else:
                    G_loss = (losses.generator_loss(D_fake))# / float(config['num_G_accumulations'])

                G_loss.backward()

                if state_dict['itr'] % 1000 == 0:# and accumulation_index == 6:
                    print('saving img')
                    # torchvision.utils.save_image(D_fake.float().cpu(),
                    #                            '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_dfake.jpg'.format(time,
                    #                                state_dict['itr'], fb_iter),
                    #                            nrow=int(D_fake.shape[0] ** 0.5),
                    #                            normalize=True)
                    torchvision.utils.save_image(G_z.float().cpu(),
                                               '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_G_z.jpg'.format(time,
                                                   state_dict['itr'], fb_iter),
                                               nrow=int(D_fake.shape[0] ** 0.5),
                                               normalize=True)
                    torchvision.utils.save_image(G_fake.float().cpu(),
                                               '/ubc/cs/research/shield/projects/cshen001/BigGAN-original/BigGAN-PyTorch/samples_new/{}_it{}_fb{}_G_z_input.jpg'.format(time,
                                                   state_dict['itr'], fb_iter),
                                               nrow=int(D_fake.shape[0] ** 0.5),
                                               normalize=True)

                #g_fakes[accumulation_index] = nn.AvgPool2d(4)(G_z).detach()
                g_fakes = nn.AvgPool2d(4)(G_z).detach()
                #d_fakes[accumulation_index] = D_fake.detach()
                d_fakes = D_fake.detach()

                # Optionally apply modified ortho reg in G
                if config['G_ortho'] > 0.0:
                    print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
                    # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
                    utils.ortho(G, config['G_ortho'],
                                      blacklist=[param for param in G.shared.parameters()])
                G.optim.step()
                    #G.optim.zero_grad()

        # If we have an ema, update it, regardless of if we test with it or not
        if config['ema']:
          ema.update(state_dict['itr'])

        out = {'G_loss': float(G_loss.item()),
                'D_loss_real': float(D_loss_real.item()),
                'D_loss_fake': float(D_loss_fake.item())}
        # Return G's loss and the components of D's loss.

        partial_test_input = partial_test_input / (inner_iter_count + 1e-9)
        return out, partial_test_input
    return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
    utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
    # Save an additional copy to mitigate accidental corruption if process
    # is killed during a save (it's happened to me before -.-)
    if config['num_save_copies'] > 0:
        utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
        state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
    # Use EMA G for samples or non-EMA?
    which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
    if config['accumulate_stats']:
        utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
    with torch.no_grad():
        if config['parallel']:
            fixed_Gz = nn.parallel.data_parallel(which_G, inputs=(fixed_z, which_G.shared(fixed_y)))
        else:
            fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
    if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
        os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
    image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'],
                                                  experiment_name,
                                                  state_dict['itr'])
    print('save image at {}'.format(image_filename))
    torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0]**0.5), normalize=True)
    # For now, every time we save, also save sample sheets
    utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
    # Also save interp sheets
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
        utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


  
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(sample, 
                                               config['num_inception_images'],
                                               num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))