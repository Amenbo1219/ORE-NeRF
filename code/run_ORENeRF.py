import os, sys
import numpy as np
import imageio
import json
import random
import time

from datasets.nerf_dataloader_ds import NeRFRayDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from helpers.run_nerf_helpers_sp_axis_ds import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
torch.set_float32_matmul_precision('high')
def random_non_zero_choice(choices):
    non_zero_choices = [x for x in choices if x != 0]
    if not non_zero_choices:
        raise ValueError("ゼロ以外の選択肢がありません。")
    return random.choice(non_zero_choices)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = inputs.reshape(-1, inputs.shape[-1])  # [N_total_samples, 3]
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        if viewdirs.ndim == 2: 
            viewdirs = viewdirs[:, None, :].expand(inputs.shape[0], inputs.shape[1], viewdirs.shape[-1])
        elif viewdirs.shape[1] != inputs.shape[1]:
            viewdirs = viewdirs[:, :1, :].expand(inputs.shape[0], inputs.shape[1], viewdirs.shape[-1])

        dirs_flat = viewdirs.reshape(-1, viewdirs.shape[-1])  # [N_total_samples, 7]
        embedded_dirs = embeddirs_fn(dirs_flat)

        assert embedded.shape[0] == embedded_dirs.shape[0], \
            f"Mismatch: embedded={embedded.shape}, dirs_flat={embedded_dirs.shape}"

        embedded = torch.cat([embedded, embedded_dirs], dim=-1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = outputs_flat.view(*inputs.shape[:-1], outputs_flat.shape[-1])
    return outputs
        

def batchify_rays(rays_flat, chunk=1024*32,K=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk],K=K, **kwargs)
            
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret if len(all_ret[k]) > 0}
    return all_ret
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0.2, far=3.14,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: Camera intrinsics matrix.
      chunk: int. Maximum number of rays to process simultaneously.
      rays: array containing ray origin, direction, and pixel coordinates.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    rays_o = rays[:, 0:3]
    rays_d = rays[:, 3:6]
    pixel_coords = rays[:, 6:8]

    if use_viewdirs:
        viewdirs = rays_d

        if c2w_staticcam is not None:
            original_viewdirs = viewdirs.clone()
            center_rays_o = rays_o.clone()  
            camera_position = c2w_staticcam[:3, 3]
            rays_o = camera_position.expand_as(rays_o)
            rays_d = original_viewdirs
            viewdirs = original_viewdirs
            
        
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        pass

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    pixel_coords = torch.reshape(pixel_coords, [-1, 2]).int()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far, pixel_coords], -1)
    
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, K=K, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, mask=None):
    H, W = hwf
    rgbs = []
    disps = []
    depths = []  
    accs = []   
    rgbs0 = []
    disps0 = []
    depths0 = []
    accs0 = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        
        rgb, disp, acc, depth, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(rgb.device).bool()
            mask_3ch = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)
            
            rgb = rgb * mask_3ch
            
            if acc is not None:
                acc = acc * mask_tensor
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())  
        accs.append(acc.cpu().numpy())      
        
        if 'rgb0' in extras:
            rgbs0.append(extras['rgb0'].cpu().numpy())
            disps0.append(extras['disp0'].cpu().numpy())
            depths0.append(extras['depth0'].cpu().numpy())
            accs0.append(extras['acc0'].cpu().numpy())
        
        if i==0:
            print(rgb.shape, disp.shape, depth.shape)
            if 'rgb0' in extras:
                print("Coarse model outputs:", extras['rgb0'].shape, extras['disp0'].shape, extras['depth0'].shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            
            depth_normalized = depths[-1] / np.max(depths[-1]) if np.max(depths[-1]) > 0 else depths[-1]
            depth8 = to8b(depth_normalized)
            depth_filename = os.path.join(savedir, '{:03d}_depth.png'.format(i))
            imageio.imwrite(depth_filename, depth8)
            
            disp_normalized = disps[-1] / np.max(disps[-1]) if np.max(disps[-1]) > 0 else disps[-1]
            disp8 = to8b(disp_normalized)
            disp_filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(disp_filename, disp8)
            

            if len(accs) > 0:  
                acc_normalized = accs[-1]
                acc8 = to8b(acc_normalized)
                acc_filename = os.path.join(savedir, '{:03d}_acc.png'.format(i))
                imageio.imwrite(acc_filename, acc8)

            if 'rgb0' in extras:
                # RGB (Coarse)
                rgb0_8 = to8b(rgbs0[-1])
                rgb0_filename = os.path.join(savedir, '{:03d}_coarse.png'.format(i))
                imageio.imwrite(rgb0_filename, rgb0_8)
                
                # Depth (Coarse)
                depth0_normalized = depths0[-1] / np.max(depths0[-1]) if np.max(depths0[-1]) > 0 else depths0[-1]
                depth0_8 = to8b(depth0_normalized)
                depth0_filename = os.path.join(savedir, '{:03d}_depth_coarse.png'.format(i))
                imageio.imwrite(depth0_filename, depth0_8)

                if len(disps0) > 0:
                    disp0_normalized = disps0[-1] / np.max(disps0[-1]) if np.max(disps0[-1]) > 0 else disps0[-1]
                    disp0_8 = to8b(disp0_normalized)
                    disp0_filename = os.path.join(savedir, '{:03d}_disp_coarse.png'.format(i))
                    imageio.imwrite(disp0_filename, disp0_8)

                if len(accs0) > 0:
                    acc0_normalized = accs0[-1]
                    acc0_8 = to8b(acc0_normalized)
                    acc0_filename = os.path.join(savedir, '{:03d}_acc_coarse.png'.format(i))
                    imageio.imwrite(acc0_filename, acc0_8)

                if 'z_std' in extras:
                    z_std_normalized = extras['z_std'].cpu().numpy() / np.max(extras['z_std'].cpu().numpy())
                    z_std_8 = to8b(z_std_normalized)
                    z_std_filename = os.path.join(savedir, '{:03d}_z_std.png'.format(i))
                    imageio.imwrite(z_std_filename, z_std_8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = np.stack(depths, 0)  
    accs = np.stack(accs, 0)      
    
    ret_dict = {
        'rgbs': rgbs, 'disps': disps, 'depths': depths, 'accs': accs
    }
    if rgbs0:
        ret_dict.update({
            'rgbs0': np.stack(rgbs0, 0),
            'disps0': np.stack(disps0, 0), 
            'depths0': np.stack(depths0, 0),
            'accs0': np.stack(accs0, 0)
        })

    return ret_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, input_dims=3)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, input_dims=3)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,geo_invariant_rgb=args.geo).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,geo_invariant_rgb=args.geo).to(device)
        grad_vars += list(model_fine.parameters())
    model = torch.nn.DataParallel(model)
    model_fine = torch.nn.DataParallel(model_fine)
    if args.amp:
        print("[Config] Compiling models with torch.compile...")
        model = torch.compile(model, backend="inductor",options={"triton.cudagraphs": False})  # AMP対応
        model_fine = torch.compile(model_fine, backend="inductor",options={"triton.cudagraphs": False})  # AMP対応
    if args.geo:
        print("[WARNING!!] Using geo-invariant RGB model.NOT USED IN RAY_O ON NERF NETWORK!")
    total_params = count_parameters(model)
    print(f"Coarse model parameters: {total_params:,}")

    if model_fine is not None:
        total_params_fine = count_parameters(model_fine)
        print(f"Fine model parameters: {total_params_fine:,}")
    print(f"Total parameters: {total_params + total_params_fine:,}")
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname


    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '.tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path,weights_only=True)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # デバイスを取得
    device = raw.device
    
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]


    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape, device=device) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.tensor(noise, device=device)  # デバイスを指定

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)


    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                K=None                
                ):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    device = ray_batch.device
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    ray_pixel_coords = ray_batch[:, 8:10]  # (i, j) ピクセル座標
    
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    
    z_vals = z_vals.expand([N_rays, N_samples])
    
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    
    raw = network_query_fn(pts, viewdirs, network_fn) 
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0, depth_map_0 = rgb_map, disp_map, acc_map, depth_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = old_sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0 
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=4, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=10, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than in depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=5, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=100, 
                        help='frequency of render_poses video saving')


    parser.add_argument("--precache", default=True,action='store_true',
                        help='pre-cache all rays in memory for faster training')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='number of workers for data loading')
    parser.add_argument("--epoch", type=int, default=100,help='number of epochs to train for')
    parser.add_argument("--geo", action='store_true',
                        help='use geo-invariant rgb model')
    parser.add_argument("--amp", action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument("--H", type=int, default=1024,
                        help='image height for rendering when render_only')
    return parser

def save_tensor_to_npz(tensor, file_path):
    tensor_data = tensor.tolist()
    np.savez(file_path, tensor_data)

def train():
    parser = config_parser()
    args = parser.parse_args()
    
    if args.dataset_type == 'synth360':
        K = None  
        near = 0.2
        far = 3.14
    
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    device_type = "cuda"
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    scaler = torch.amp.GradScaler(device=device_type, enabled=args.amp)

    if args.render_only:
        print('RENDER ONLY')
        dataset = NeRFRayDataset(args.datadir, 'test',img_size=(args.H, args.H))
        H, W = dataset.H, dataset.W
        K = dataset.K
        mask = dataset.mask if hasattr(dataset, 'mask') else None
        render_poses = torch.Tensor(dataset.render_poses).to(device)
        
        
        with torch.no_grad():
            testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_path_{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            
            if args.render_test:
                test_dataset = NeRFRayDataset(args.datadir, 'test',img_size=(args.H, args.H))
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
                test_poses = [batch['pose'][0].to(device) for batch in test_loader]
                test_images = [batch['image'][0].cpu().numpy() for batch in test_loader]
                render_poses = torch.stack(test_poses, dim=0)
                images = np.stack(test_images, axis=0)
            else:
                images = None

            testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            rgbs = []
            disps = []
            depths = []  
            accs = []   

            
            for pose_idx in tqdm(range(len(render_poses)), desc='Rendering video'):
                ray_data = dataset.get_render_rays(pose_idx)
                rays_o = ray_data['rays_o'].to(device)  # [H, W, 3]
                rays_d = ray_data['rays_d'].to(device)  # [H, W, 3]
                pixel_coords = ray_data['pixel_coords'].to(device)  # [H, W, 2]
                rays_o_flat = rays_o.reshape(-1, 3)  # [H*W, 3]
                rays_d_flat = rays_d.reshape(-1, 3)  # [H*W, 3]
                pixel_coords_flat = pixel_coords.reshape(-1, 2)  # [H*W, 2]
                
                batch_rays = torch.cat([rays_o_flat, rays_d_flat, pixel_coords_flat], dim=-1)  # [H*W, 8]
                with torch.amp.autocast(device_type=device_type,enabled=args.amp):
                    rgb, disp, acc, depth, extras = render(
                        H, W, K, chunk=args.chunk, 
                        rays=batch_rays, c2w=dataset.K,
                        **render_kwargs_test
                    )
                
                if mask is not None:
                    mask_tensor = torch.from_numpy(mask).to(rgb.device).bool()
                    mask_3ch = mask_tensor.unsqueeze(-1).reshape(H,W,1).repeat(1, 1, 3)
                    rgb = rgb.reshape(H, W, 3)
                    rgb = rgb * mask_3ch
                    
                    if acc is not None:
                        acc = acc.reshape(H, W)
                        acc = acc * mask_tensor.reshape(H,W)
                        
                    disp = disp.reshape(H, W)
                    depth = depth.reshape(H, W)
                else:
                    rgb = rgb.reshape(H, W, 3)
                    disp = disp.reshape(H, W)
                    depth = depth.reshape(H, W)
                    acc = acc.reshape(H, W)
                
                rgbs.append(rgb.cpu().numpy())
                disps.append(disp.cpu().numpy())
                depths.append(depth.cpu().numpy())  
                accs.append(acc.cpu().numpy())  
            
            rgbs = np.stack(rgbs, 0)
            disps = np.stack(disps, 0)
            depths = np.stack(depths, 0)  
            accs = np.stack(accs, 0)  
            
            print('Done rendering video, saving', rgbs.shape, disps.shape, depths.shape)
            moviebase = os.path.join(args.basedir,args.expname,)
            
            imageio.mimwrite(os.path.join(moviebase, 'rgb.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(moviebase, 'disp.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)
            imageio.mimwrite(os.path.join(moviebase, 'depth.mp4'), to8b(depths / np.max(depths)), fps=30, quality=8)

            return

    train_dataset = NeRFRayDataset(
        args.datadir, 
        mode='train', 
        rays_per_image=args.N_rand, 
        precache=args.precache,
        img_size=(args.H, args.H)
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True,  
        num_workers=0  
    )

    test_dataset = NeRFRayDataset(args.datadir, mode='test',img_size=(args.H, args.H))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    mask = train_dataset.mask if hasattr(train_dataset, 'mask') else None
    H, W = train_dataset.H, train_dataset.W
    K = train_dataset.K
    render_poses = torch.Tensor(train_dataset.render_poses).to(device)
    
    print('Begin')
    print(f'Dataset: train={len(train_dataset)}, test={len(test_dataset)}')
    print(f'Image size: {H}x{W}')
    print(f'Camera intrinsic: K={K}')
    print(f'Rays per batch: {args.N_rand}')
    Each_epoch_N_iters = (train_dataset.total_valid_rays // args.N_rand)+1
    Epoch = args.epoch
    N_iters = Each_epoch_N_iters * Epoch
    train_iter = iter(train_loader)
    RATIO_DECAY_START  = N_iters * 0.3   
    pbar = trange(start, N_iters+1, unit='step')
    for i in pbar:
        current_epoch = i // Each_epoch_N_iters
        step_in_epoch = i % Each_epoch_N_iters
        pbar.set_postfix({'Epoch': current_epoch, 'Step': step_in_epoch})
        time0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        rays_o = batch['rays_o'][0].to(device)  # [N_RAND, 3]
        rays_d = batch['rays_d'][0].to(device)  # [N_RAND, 3]
        pixel_coords = batch['pixel_coords'][0].to(device)  # [N_RAND, 2]
        target_rgb = batch['target_rgb'][0].to(device)  # [N_RAND, 3]
        batch_mask = batch['mask'][0] if batch['mask'][0] is not None else None
        cam_key = batch['cam_key'][0] if 'cam_key' in batch else None
        batch_rays = torch.cat([rays_o, rays_d, pixel_coords], dim=-1)  # [N_RAND, 8]
        with torch.amp.autocast(device_type=device_type, enabled=args.amp):
            rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True, **render_kwargs_train)
            optimizer.zero_grad()
            if batch_mask is not None:
                pixel_i = pixel_coords[:, 0].long()  
                pixel_j = pixel_coords[:, 1].long()  
                batch_mask_gpu = batch_mask.to(device)
                mask_values = batch_mask_gpu[pixel_j, pixel_i]  # [N_RAND] - ピクセル座標でマスク値を取得
                valid_pixels = mask_values.bool()
        
                if valid_pixels.sum() > 0:
                    img_loss = img2mse(rgb[valid_pixels], target_rgb[valid_pixels])
                    loss = img_loss
                    psnr = mse2psnr(img_loss)
                    
                    if 'rgb0' in extras:
                        img_loss0 = img2mse(extras['rgb0'][valid_pixels], target_rgb[valid_pixels])
                        loss = loss + img_loss0
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    psnr = torch.tensor(0.0, device=device)
            else:
                img_loss = img2mse(rgb, target_rgb)
                loss = img_loss
                psnr = mse2psnr(img_loss)
                if 'rgb0' in extras:
                    img_loss0 = img2mse(extras['rgb0'], target_rgb)
                    loss = loss + img_loss0
                    psnr0 = mse2psnr(img_loss0)  
            initial_lr = args.lrate
            decay_rate = 0.5
            
            if i <=  RATIO_DECAY_START:
                None
            else:
                new_lrate = args.lrate * (decay_rate ** (global_step / N_iters))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
            if args.amp:
                scaler.scale(loss).backward()  
                scaler.step(optimizer)         
                scaler.update()                
            else:
                loss.backward()
                optimizer.step()

        dt = time.time() - time0
        
        if i % (Each_epoch_N_iters * args.i_weights)==0 and i > 0:
            path = os.path.join(args.basedir, args.expname, '{:09d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
        if i % (Each_epoch_N_iters * args.i_video)==0 and i > 0:
            print(f'\n[Video] Rendering video at iteration {i}...')
            with torch.no_grad():
                rgbs = []
                disps = []
                depths = [] 
                accs = []   
                rgbs0 = []
                disps0 = []
                depths0 = []
                accs0 = []
                
                for pose_idx in tqdm(range(len(render_poses)), desc='Rendering video'):
                    ray_data = train_dataset.get_render_rays(pose_idx)
                    
                    rays_o = ray_data['rays_o'].to(device)  # [H, W, 3]
                    rays_d = ray_data['rays_d'].to(device)  # [H, W, 3]
                    pixel_coords = ray_data['pixel_coords'].to(device)  # [H, W, 2]
                    c2w = ray_data['c2w'].to(device)  # [4, 4]
                    
                    rays_o_flat = rays_o.reshape(-1, 3)  # [H*W, 3]
                    rays_d_flat = rays_d.reshape(-1, 3)  # [H*W, 3]
                    pixel_coords_flat = pixel_coords.reshape(-1, 2)  # [H*W, 2]
                    
                    batch_rays = torch.cat([rays_o_flat, rays_d_flat, pixel_coords_flat], dim=-1)  # [H*W, 8]
                    with torch.amp.autocast(device_type=device_type,enabled=args.amp):
                        rgb, disp, acc, depth, extras = render(
                            H, W, K, chunk=args.chunk, 
                            rays=batch_rays, c2w=c2w,
                            **render_kwargs_test
                        )
                    
                    if mask is not None:
                        mask_tensor = torch.from_numpy(mask).to(rgb.device).bool()
                        mask_3ch = mask_tensor.unsqueeze(-1).reshape(H,W,1).repeat(1, 1, 3)
                        rgb = rgb.reshape(H, W, 3)
                        rgb = rgb * mask_3ch
                        if acc is not None:
                            acc = acc.reshape(H, W)
                            acc = acc * mask_tensor.reshape(H,W)
                            
                        disp = disp.reshape(H, W)
                        depth = depth.reshape(H, W)
                    else:
                        rgb = rgb.reshape(H, W, 3)
                        disp = disp.reshape(H, W)
                        depth = depth.reshape(H, W)
                        acc = acc.reshape(H, W)
                    
                    rgbs.append(rgb.cpu().numpy())
                    disps.append(disp.cpu().numpy())
                    depths.append(depth.cpu().numpy())  
                    accs.append(acc.cpu().numpy()) 
                    
                    if 'rgb0' in extras:
                        rgb0 = extras['rgb0'].reshape(H, W, 3)
                        disp0 = extras['disp0'].reshape(H, W)
                        depth0 = extras['depth0'].reshape(H, W)
                        acc0 = extras['acc0'].reshape(H, W)
                        
                        rgbs0.append(rgb0.cpu().numpy())
                        disps0.append(disp0.cpu().numpy())
                        depths0.append(depth0.cpu().numpy())
                        accs0.append(acc0.cpu().numpy())
                
                rgbs = np.stack(rgbs, 0)
                disps = np.stack(disps, 0)
                depths = np.stack(depths, 0)  
                accs = np.stack(accs, 0)  
            print('Done rendering video, saving', rgbs.shape, disps.shape, depths.shape)
            moviebase = os.path.join(args.basedir, args.expname, '{}_spiral_{:06d}_'.format(args.expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', to8b(depths / np.max(depths)), fps=30, quality=8)
            if len(rgbs0) > 0:
                rgbs0 = np.stack(rgbs0, 0)
                disps0 = np.stack(disps0, 0)
                depths0 = np.stack(depths0, 0)
                
                print('Saving coarse model results as well:', rgbs0.shape, disps0.shape, depths0.shape)
                imageio.mimwrite(moviebase + 'rgb_coarse.mp4', to8b(rgbs0), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp_coarse.mp4', to8b(disps0 / np.max(disps0)), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'depth_coarse.mp4', to8b(depths0 / np.max(depths0)), fps=30, quality=8)
            print('Video saved at', moviebase)
        if i % (Each_epoch_N_iters * args.i_testset) == 0 and i > 0:
            testsavedir = os.path.join(args.basedir, args.expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Rendering test set"):
                    rays_o = batch['rays_o'][0].to(device)  # [N_rays, 3]
                    rays_d = batch['rays_d'][0].to(device)  # [N_rays, 3]
                    pixel_coords = batch['pixel_coords'][0].to(device)  # [N_rays, 2]
                    target_rgb = batch['image'][0].cpu().numpy()  # [H, W, 3] (GT画像)
                    batch_mask = batch['mask'][0] if batch['mask'][0] is not None else None
                    cam_key = batch['cam_key'][0] if 'cam_key' in batch else None
                    batch_rays = torch.cat([rays_o, rays_d, pixel_coords], dim=-1)  # [W, H, 8]
                    batch_rays = batch_rays.view(-1, 8)  # # Reshape to [N_rays, 8] where N_rays = W * H
                    rgb, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, **render_kwargs_test)
                    rgb = rgb.view(H, W, 3)  # or rgb.reshape(H, W, 3)
                    depth = depth.view(H, W)
                    disp = disp.view(H, W)
                    acc = acc.view(H, W)
                    if batch_mask is not None:
                        mask_tensor = batch_mask.to(rgb.device).bool()
                        mask_3ch = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)
                        rgb = rgb * mask_3ch
                        acc = acc * mask_tensor
                        depth = depth * mask_tensor
                        disp = disp * mask_tensor
                    rgb8 = to8b(rgb.cpu().numpy())
                    filename = os.path.join(testsavedir, '{:03d}.png'.format(idx))
                    imageio.imwrite(filename, rgb8)
                    if cam_key:
                        cam_filename = os.path.join(testsavedir, '{:03d}_cam_{}.txt'.format(idx, cam_key))
                        with open(cam_filename, 'w') as f:
                            f.write(f"Camera: {cam_key}\n")
                    if batch_mask is not None:
                        mask8 = to8b(batch_mask.cpu().numpy().astype(np.float32))
                        mask_filename = os.path.join(testsavedir, '{:03d}_mask.png'.format(idx))
                        imageio.imwrite(mask_filename, mask8)
                    depth_max = torch.max(depth).cpu().numpy()
                    depth_normalized = depth.cpu().numpy() / depth_max if depth_max > 0 else depth.cpu().numpy()
                    depth8 = to8b(depth_normalized)
                    depth_filename = os.path.join(testsavedir, '{:03d}_depth.png'.format(idx))
                    imageio.imwrite(depth_filename, depth8)
                    disp_normalized = disp.cpu().numpy() / torch.max(disp).cpu().numpy()
                    disp8 = to8b(disp_normalized)
                    disp_filename = os.path.join(testsavedir, '{:03d}_disp.png'.format(idx))
                    imageio.imwrite(disp_filename, disp8)
                    acc_normalized = acc.cpu().numpy()
                    acc8 = to8b(acc_normalized)
                    acc_filename = os.path.join(testsavedir, '{:03d}_acc.png'.format(idx))
                    imageio.imwrite(acc_filename, acc8)
                    rgb_filename = os.path.join(testsavedir, '{:03d}-rgb.npy'.format(idx))
                    np.save(rgb_filename, rgb.cpu().numpy())
                    gt_filename = os.path.join(testsavedir, '{:03d}-gt.npy'.format(idx))
                    np.save(gt_filename, target_rgb)
                    if 'rgb0' in extras:
                        rgb0 = extras['rgb0'].view(H, W, 3)
                        depth0 = extras['depth0'].view(H, W)
                        disp0 = extras['disp0'].view(H, W)
                        acc0 = extras['acc0'].view(H, W)
                        rgb0_8 = to8b(rgb0.cpu().numpy())
                        rgb0_filename = os.path.join(testsavedir, '{:03d}_coarse.png'.format(idx))
                        imageio.imwrite(rgb0_filename, rgb0_8)
                        depth0_normalized = depth0.cpu().numpy() / torch.max(depth0).cpu().numpy()
                        depth0_8 = to8b(depth0_normalized)
                        depth0_filename = os.path.join(testsavedir, '{:03d}_depth_coarse.png'.format(idx))
                        imageio.imwrite(depth0_filename, depth0_8)
                        disp0_normalized = disp0.cpu().numpy() / torch.max(disp0).cpu().numpy()
                        disp0_8 = to8b(disp0_normalized)
                        disp0_filename = os.path.join(testsavedir, '{:03d}_disp_coarse.png'.format(idx))
                        imageio.imwrite(disp0_filename, disp0_8)
                        acc0_normalized = acc0.cpu().numpy()
                        acc0_8 = to8b(acc0_normalized)
                        acc0_filename = os.path.join(testsavedir, '{:03d}_acc_coarse.png'.format(idx))
                        imageio.imwrite(acc0_filename, acc0_8)
                        if 'z_std' in extras:
                            z_std = extras['z_std'].view(H, W)
                            z_std_normalized = z_std.cpu().numpy() / torch.max(z_std).cpu().numpy()
                            z_std_8 = to8b(z_std_normalized)
                            z_std_filename = os.path.join(testsavedir, '{:03d}_z_std.png'.format(idx))
                            imageio.imwrite(z_std_filename, z_std_8)
                print('Saved test set')
        if i % (Each_epoch_N_iters * args.i_print) == 0 :
            tqdm.write(f"[TRAIN] Epoch: {current_epoch}/{Epoch} Step: {i} ({step_in_epoch}/{Each_epoch_N_iters}) Loss: {loss.item():.6f} PSNR: {psnr.item():.2f}")
            
        global_step += 1


if __name__=='__main__':
    train()
