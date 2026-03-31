import os
import numpy as np
import cv2
import torch
import math
import random
from torch.utils.data import Dataset, DataLoader
# from scipy.interpolate import interp1d
import pandas as pd
import os 
import csv
class NeRFRayDataset(Dataset):
    def __init__(self, datadir, mode='train', rays_per_image=1024, transform=None, K=None, precache=False,img_size=(1024, 1024)):
        """
        datadir : データフォルダ（例：synth360形式なら train/と test/がある）
        mode    : 'train' / 'test' を指定
        rays_per_image : 各画像から抽出するレイの数（mode='train'時のみ使用）
        transform : 必要に応じた前処理関数
        K       : カメラ内部パラメータ。Noneの場合はデフォルト値を用いる
        precache: 画像とレイをあらかじめメモリ上にキャッシュするかどうか
        """
        self.datadir = datadir
        self.mode = mode
        self.transform = transform
        self.rays_per_image = rays_per_image
        self.precache = precache
        self.img_size = img_size
        self.pose_normalization = {}
        self._raw_scene_bounds = self.load_scene_bbox(datadir)  # シーンの境界ボックスを読み込む
        self.scene_bounds = [-np.pi,np.pi,-np.pi,np.pi,-np.pi,np.pi]  # デフォルトの境界ボックス（後で更新される）
        
        # self.distortion_df = pd.read_csv(os.path.join(self.datadir,"distortion.csv"))
        # マスク画像の読み込み
        mask_path = os.path.join(datadir, 'mask.png')
        if os.path.exists(mask_path):
            # まずリサイズしてから二値化
            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_resized = cv2.resize(mask_gray, self.img_size)  # 指定サイズにリサイズ  # 先にリサイズ
            self.mask = mask_resized > 0  # 後で二値化
            print(f"Loaded mask from {mask_path}, valid pixels: {np.sum(self.mask)}/{self.mask.size}")
        else:
            print(f"Warning: Mask file not found at {mask_path}. Using all pixels.")
            self.mask = None
        
        # ファイルパスとポーズ情報だけを先に読み込む
        self.train_img_paths, self.train_poses = self._load_paths_and_poses(os.path.join(datadir, 'train'))
        self.test_img_paths, self.test_poses = self._load_paths_and_poses(os.path.join(datadir, 'test'))
        # ポーズの正規化（全てのポーズをまとめて正規化）
        self.raw_all_poses = np.concatenate([self.train_poses, self.test_poses], axis=0)
        self.raw_all_poses, self._raw_scene_bounds = self.pose_trance(self.raw_all_poses,self._raw_scene_bounds)
        print(self.raw_all_poses[0])
        # all_poses = all_poses @ np.diag([1, 1, -1, 1]).astype(np.float32)  # OpenCVからOpenGL座標系に変換
        # self.normalized_poses = self.normalize_poses_translation_to_01(all_poses.copy())
        # # BBox正規化（flip_z=Trueで同じ座標系に変換）
        self.normalized_scene_bounds, self.pose_normalization = self.normalize_bbox(self._raw_scene_bounds, flip_z=False)
        self.normalized_poses = self.normalize_poses(self.raw_all_poses)
        # print(self.normalized_poses)
        
        # # 正規化後のBboxを使用
        self.scene_bounds = self.normalized_scene_bounds
        # 画像のサイズを取得するために1枚だけ読み込む
        sample_img = self._load_image(self.train_img_paths[0])
        self.H, self.W = sample_img.shape[:2]
        # カメラ内部パラメータをセット
        if K is None:
            f_cm = 2.57 # 焦点距離（cm）
            X_mm = 8.7552 # センサーサイズ（mm）
            p_mm = X_mm / self.W # ピクセルサイズ（mm）
            f_eq = f_cm / p_mm # 焦点距離（ピクセル） 
            self.K = np.array([[f_eq, 0, 0.5 * self.W],
                               [0, f_eq, 0.5 * self.H],
                               [0, 0, 1]], dtype=np.float32)
        else:
            self.K = K
        
        # レンダリングパス生成（動画作成時などに使用）
        self.render_poses = self.interpolate_poses(self.normalized_poses).astype(np.float32)
        
        # モードに応じて画像パスとポーズを設定
        if mode == 'train':
            self.img_paths = self.train_img_paths
            self.poses = self.normalized_poses[:len(self.train_img_paths)]
        elif mode == 'test':
            self.img_paths = self.test_img_paths
            self.poses = self.normalized_poses[len(self.train_img_paths):]
        else:  # 'all'
            self.img_paths = self.train_img_paths + self.test_img_paths
            self.poses = self.normalized_poses
        
        # 有効なピクセル座標のインデックスを計算
        if self.mask is not None:
            self.valid_indices = np.where(self.mask.flatten())[0]
            print(f"Valid indices for sampling: {len(self.valid_indices)}")
        
        # プリキャッシュ（オプション）
        self.cached_data = None
        if precache:
            self._cache_all_rays()
    @property
    def total_valid_rays(self):
        """総Rayの数を返す（マスク関係なし）"""
        return self.H * self.W * len(self.train_img_paths)
    
    @property
    def valid_pixels_per_image(self):
        """1画像あたりの総ピクセル数を返す（マスク関係なし）"""
        return self.H * self.W
    def _cache_all_rays(self):
        """全画像のレイをキャッシュ（学習時）またはテスト画像をロード（評価時）"""
        self.cached_data = []
        
        for i, (img_path, pose) in enumerate(zip(self.img_paths, self.poses)):
            # 画像読み込み
            img = self._load_image(img_path)
            
            # マスク適用（オプション）
            if self.mask is not None:
                mask_3ch = np.stack([self.mask] * 3, axis=-1)  # 3チャンネルに拡張
                img_masked = img * mask_3ch  # マスク外を0（黒）に
            else:
                img_masked = img
            
            # レイ計算
            rays_o, rays_d, pixel_coords = self.get_rays_np_fisyeye(self.H, self.W, self.K, pose)
            # rays_o, rays_d, pixel_coords = self.get_rays_theta_z1(self.H,self.W, pose,self.distortion_df)
            
            if self.mode == 'train':
                # 学習モード：ピクセル座標を平坦化してキャッシュ
                self.cached_data.append({
                    'image': torch.from_numpy(img_masked.copy()).float(),
                    'pose': torch.from_numpy(pose.copy()).float(),
                    'rays_o': torch.from_numpy(rays_o.copy()).float(),
                    'rays_d': torch.from_numpy(rays_d.copy()).float(),
                    'pixel_coords': torch.from_numpy(pixel_coords.copy()).float(),
                    'mask': torch.from_numpy(self.mask).bool() if self.mask is not None else None
                })
            else:
                # テストモード：画像とレイ全体をキャッシュ
                self.cached_data.append({
                    'image': torch.from_numpy(img_masked.copy()).float(),          # [H, W, 3]
                    'pose': torch.from_numpy(pose.copy()).float(),                 # [4, 4]
                    'rays_o': torch.from_numpy(rays_o.copy()).float(),             # [H, W, 3] 
                    'rays_d': torch.from_numpy(rays_d.copy()).float(),             # [H, W, 3]
                    'pixel_coords': torch.from_numpy(pixel_coords.copy()).float(), # [H, W, 2]
                    'mask': torch.from_numpy(self.mask).bool() if self.mask is not None else None  # [H, W]
                })
    def load_scene_bbox(self,basedir,csv_filename: str = "scene_bbox.csv") -> dict:
        """
        project_root/scene_bbox.csv を読み込み、Parameter→Value の dict を返す
        """
        # このファイルから見た project_root へのパス
        csv_path = os.path.join(basedir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            print("Using default scene bounds: [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]")
            return [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]
        else:
            bounds = {}    
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    bounds[row["Parameter"]] = float(row["Value"])
            # 元のOpenCV座標系のまま返す（座標系変換は後で統一して行う）
            result = [bounds["x0"], bounds["x1"], bounds["y0"], bounds["y1"], bounds["z0"], bounds["z1"]]
            print(f"Loaded scene bounds from CSV (OpenCV coords): {result}")
            return result
    def normalize_poses(self, poses):
        """
        BBox由来のcenter/scaleでPoseを正規化する
        注意: 呼び出し側で事前にOpenCV→OpenGL変換が済んでいる前提
        """
        assert hasattr(self, "center") and hasattr(self, "scale"), \
            "Call normalize_bbox() before normalize_poses()"

        poses_norm = poses.copy()
        poses_norm[:, :3, 3] = (poses[:, :3, 3] - self.center) / self.scale

        
        print("[Pose Normalization] applied from BBox center/scale.")
        return poses_norm
    def normalize_bbox(self, bbox, flip_z=False,target_range=1.0):
        """
        シーン全体のAABB (bbox) から正規化の中心とスケールを計算し、
        さらにBBoxを[-1,1]相当の空間にスケーリングして返す。
        
        Args:
            bbox: list or np.ndarray [xmin, xmax, ymin, ymax, zmin, zmax]
            flip_z: bool = True の場合、OpenGL座標系に合わせてZ軸を反転
        
        Returns:
            normalized_bbox: np.ndarray [xmin, xmax, ymin, ymax, zmin, zmax]
            pose_normalization: dict {"center": np.array([3]), "scale": float}
        """
        bbox = np.array(bbox, dtype=np.float32)
        assert bbox.size == 6, f"Expected 6 values, got {bbox.size}"

        # --- min/max から中心とスケールを算出 ---
        bbox_min = np.array([bbox[0], bbox[2], bbox[4]], dtype=np.float32)
        bbox_max = np.array([bbox[1], bbox[3], bbox[5]], dtype=np.float32)
        
        # # Z軸反転を考慮してcenterとscaleを計算
        # if flip_z:
        #     # Z座標を反転
        #     bbox_min_transformed = bbox_min.copy()
        #     bbox_max_transformed = bbox_max.copy()
        #     bbox_min_transformed[2] *= -1.0
        #     bbox_max_transformed[2] *= -1.0
        #     # 反転後のmin/maxを再計算
        #     z_min = min(bbox_min_transformed[2], bbox_max_transformed[2])
        #     z_max = max(bbox_min_transformed[2], bbox_max_transformed[2])
        #     bbox_min_transformed[2] = z_min
        #     bbox_max_transformed[2] = z_max
        # else:
        bbox_min_transformed = bbox_min
        bbox_max_transformed = bbox_max
            
        center = (bbox_min_transformed + bbox_max_transformed) / 2.0
        diag = bbox_max_transformed - bbox_min_transformed
        scale = np.max(diag) / (2.0 * float(target_range)) 
        # --- 正規化（中心化＋スケール） ---
        norm_bbox_min = (bbox_min_transformed - center) / scale
        norm_bbox_max = (bbox_max_transformed - center) / scale

        # --- min/max順を保証 ---
        norm_bbox_min_final = np.minimum(norm_bbox_min, norm_bbox_max)
        norm_bbox_max_final = np.maximum(norm_bbox_min, norm_bbox_max)

        normalized_bbox = np.array([
            norm_bbox_min_final[0], norm_bbox_max_final[0],
            norm_bbox_min_final[1], norm_bbox_max_final[1],
            norm_bbox_min_final[2], norm_bbox_max_final[2],
        ], dtype=np.float32)

        # --- 保存（Pose正規化でも使うため） ---
        self.center = center
        self.scale = scale
        self.flip_z = flip_z

        pose_normalization = {"center": center, "scale": scale}

        # print("[BBox Normalization]")
        # print(f"  Raw bbox     : {np.round(bbox, 4)}")
        # print(f"  Center       : {np.round(center, 4)}")
        # print(f"  Scale (half) : {scale:.4f}")
        # print(f"  Flip Z       : {flip_z}")
        print(f"  Normalized   : {np.round(normalized_bbox, 4)}")

        return normalized_bbox, pose_normalization
    def __len__(self):
        """
        学習モード：バッチ数 (画像数 * rays_per_image / バッチサイズ に相当)
        テストモード：画像数（テスト時は1画像ずつ処理）
        """
        if self.mode == 'train':
            # 十分な学習ステップのため多めに設定：例えば40万ステップ
            return 400000  # イテレーション数（任意に調整可能）
        else:
            return len(self.img_paths)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # 学習モード：ランダムな画像から rays_per_image の数だけレイをサンプリング
            img_idx = random.randint(0, len(self.img_paths) - 1)
            
            if self.precache and self.cached_data is not None:
                # キャッシュから読み込み
                img = self.cached_data[img_idx]['image']
                rays_o = self.cached_data[img_idx]['rays_o']
                rays_d = self.cached_data[img_idx]['rays_d']
                pixel_coords = self.cached_data[img_idx]['pixel_coords']
            else:
                # 画像を読み込み
                img_path = self.img_paths[img_idx]
                pose = self.poses[img_idx]
                img = torch.from_numpy(self._load_image(img_path).copy()).float()
                
                # レイ計算
                rays_o, rays_d, pixel_coords = self.get_rays_np_fisyeye(self.H, self.W, self.K, pose)
                # rays_o, rays_d, pixel_coords = self.get_rays_theta_z1(self.H, self.W,pose,self.distortion_df)
                rays_o = torch.from_numpy(rays_o.copy()).float()
                rays_d = torch.from_numpy(rays_d.copy()).float()
                pixel_coords = torch.from_numpy(pixel_coords.copy()).float()
                # mask_tensor = torch.tensor(self.mask, dtype=torch.bool, device=rays_o.device)
                
                # near, far = self.compute_near_far_from_aabb(
                #     rays_o, 
                #     rays_d, 
                #     self.scene_bounds, 
                # mask_tensor)
            # マスクを考慮してサンプリング
            h, w = img.shape[:2]
            
            if self.mask is not None:
                # マスク内のピクセルからのみサンプリング
                if len(self.valid_indices) < self.rays_per_image:
                    # 有効なピクセルが少ない場合は重複を許可
                    select_flat_idxs = np.random.choice(self.valid_indices, size=self.rays_per_image, replace=True)
                else:
                    # 十分な有効ピクセルがある場合は重複なし
                    select_flat_idxs = np.random.choice(self.valid_indices, size=self.rays_per_image, replace=False)
                
                # 1次元インデックスを2次元(h,w)に変換
                select_inds_h = select_flat_idxs // w
                select_inds_w = select_flat_idxs % w
            else:
                # マスクなしの場合は全ピクセルから均等サンプリング
                select_inds = torch.randint(0, h * w, size=(self.rays_per_image,))
                select_inds_h = select_inds // w
                select_inds_w = select_inds % w
            
            # サンプリングされたピクセルに対応するレイ情報を取得
            rays_o_sampled = rays_o[select_inds_h, select_inds_w]  # [rays_per_image, 3]
            rays_d_sampled = rays_d[select_inds_h, select_inds_w]  # [rays_per_image, 3]
            pixel_coords_sampled = pixel_coords[select_inds_h, select_inds_w]  # [rays_per_image, 2]
            target_rgb = img[select_inds_h, select_inds_w]  # [rays_per_image, 3]
            # near, far = self.compute_near_far_from_aabb(rays_o_sampled, rays_d_sampled, self.scene_bounds, mask=self.mask)


            # サンプルのバッチを返す
            sample = {
                'rays_o': rays_o_sampled,                   # [rays_per_image, 3]
                'rays_d': rays_d_sampled,                   # [rays_per_image, 3]
                'pixel_coords': pixel_coords_sampled,       # [rays_per_image, 2]
                'target_rgb': target_rgb,                   # [rays_per_image, 3]
                'image_idx': torch.tensor(img_idx, dtype=torch.int32),  # 画像インデックス
                'mask': torch.from_numpy(self.mask).bool() if self.mask is not None else None,  # [H, W]
            }
            
        else:
            # テストモード：インデックスの画像に対応するレイを全て返す
            img_idx = idx % len(self.img_paths)
            
            if self.precache and self.cached_data is not None:
                # キャッシュから読み込み
                sample = self.cached_data[img_idx]
                
                # マスク適用（マスクの無効なピクセルは透明に設定）
                if self.mask is not None and 'image' in sample:
                    img = sample['image'].clone()  # コピーを作成
                    mask_tensor = torch.from_numpy(self.mask).bool()
                    # RGB画像の場合は3チャンネル全てに適用
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img[~mask_tensor] = 0.0  # マスク外を透明に
                    sample['image'] = img
                    sample['mask'] = mask_tensor  # マスク情報も追加
                
                return sample
            else:
                # 画像を読み込み
                img_path = self.img_paths[img_idx]
                pose = self.poses[img_idx]
                img = self._load_image(img_path)
                
                # マスク適用
                if self.mask is not None:
                    mask_3ch = np.stack([self.mask] * 3, axis=-1)  # 3チャンネルに拡張
                    img = img * mask_3ch  # マスク外を0（黒）に
                
                # レイ計算
                mask_tensor = torch.tensor(self.mask, dtype=torch.bool) if self.mask is not None else None
                rays_o, rays_d, pixel_coords = self.get_rays_np_fisyeye(self.H, self.W, self.K, pose)
                # rays_o, rays_d, pixel_coords = self.get_rays_theta_z1(self.H, self.W,pose,self.distor /tion_df)
                # near, far = self.compute_near_far_from_aabb(rays_o, rays_d, self.scene_bounds, mask=mask_tensor)
                # サンプルを返す
                sample = {
                    'image': torch.from_numpy(img.copy()).float(),           # [H, W, 3]
                    'pose': torch.from_numpy(pose.copy()).float(),           # [4, 4]
                    'rays_o': torch.from_numpy(rays_o.copy()).float(),       # [H, W, 3]
                    'rays_d': torch.from_numpy(rays_d.copy()).float(),       # [H, W, 3]
                    'pixel_coords': torch.from_numpy(pixel_coords.copy()).float(),  # [H, W, 2]
                    'mask': torch.from_numpy(self.mask).bool() if self.mask is not None else None,  # [H, W]
                    # 'near': torch.tensor(near, dtype=torch.float32),         # scalar
                    # 'far': torch.tensor(far, dtype=torch.float32),           # scalar
                }
                        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _load_paths_and_poses(self, basedir):
        imgdir = os.path.join(basedir, 'images')
        
        img_paths = []
        poses = []
        
        with open(os.path.join(basedir, 'poses.txt')) as f:
            for line in f.readlines():
                line = line.rstrip()
                line = line.split(" ")
                img_name = line[0] + ".png"
                img_path = os.path.join(imgdir, img_name)
                pose = self.transform_pose(line)
                
                img_paths.append(img_path)
                poses.append(pose)
        
        return img_paths, np.array(poses, dtype=np.float32)
    
    def _load_image(self, img_path):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
        img = cv2.resize(img, self.img_size)  # 指定サイズにリサイズ
        return img.astype(np.float32)
    



    def get_rays_np_fisyeye(self, H, W, K, c2w):

        i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                          np.arange(H, dtype=np.float32), indexing='xy')
        # マスクがある場合、マスク領域外のピクセルは特別な値を設定
        if hasattr(self, 'mask') and self.mask is not None:
            # マスク適用
            valid_pixels = self.mask  # ブール型マスク [H, W]
            valid_indices = np.argwhere(self.mask)  # [[j, i], ...]
            cy, cx = valid_indices.mean(axis=0)  # マスク重心を中心として設定
            # 画像中心からの距離を計算
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

            # 球面座標系のためのphi角度 (方位角)
            phi = np.arctan2(j - cy, i - cx)
            # --- マスクされた範囲内のrの最大値を取得 ---
            r_valid = r[valid_pixels]
            r_max = r_valid.max()
            # --- thetaの再スケーリング ---
            # ここでr_maxがfisheyeの設計視野角に相当するようにスケーリングする
            # 例：fisheye視野角 = 180° -> θ_max = π/2
            fisheye_fov_deg = 180.0  # 190度のFOV
            # fisheye_fov_deg = 180.0  # 190度のFOV
            
            fisheye_fov_rad = np.radians(fisheye_fov_deg / 2)  # 190° fisheyeなら95°をラジアンに


            theta = np.zeros_like(r)
            theta[valid_pixels] = (r_valid / r_max) * fisheye_fov_rad
            
            # --- phi の計算 ---
            # phi_valid = phi[valid_pixels]
            # phi_min = phi_valid.min()
            # phi_max = phi_valid.max()
            # phi_scaled = np.zeros_like(phi)
            # phi_scaled[valid_pixels] = (phi_valid - phi_min) / (phi_max - phi_min + 1e-6)
            # phi_scaled = phi_scaled * 2 * np.pi  # 周期性を保った方位角に変換

            # --- 光線方向ベクトル計算 ---
            x = np.zeros_like(r)
            y = np.zeros_like(r)
            z = np.zeros_like(r)
            x[valid_pixels] = np.sin(theta[valid_pixels]) * np.cos(phi[valid_pixels])
            y[valid_pixels] = np.sin(theta[valid_pixels]) * np.sin(phi[valid_pixels])
            z[valid_pixels] = np.cos(theta[valid_pixels])
            
        else:
            cx, cy = W / 2, H / 2
            f_eq = K[0][0]
            # 画像中心からの距離を計算
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

            # 球面座標系のためのphi角度 (方位角)
            phi = np.arctan2(j - cy, i - cx)
            # マスクなしの場合、全ピクセルを処理
            theta = r / f_eq  # θ (天頂角) [H, W]
            
            # 光線方向の各成分を計算（球面座標系を使用）
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            

        # レイ方向ベクトル（カメラ座標系）- y軸を反転
        dirs = np.stack([x, -y, -z], -1)  # [H, W, 3]
        # dirs = np.stack([x, -y, z], -1)  # [H, W, 3]
 
        # ワールド座標系に変換
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # 回転行列適用
        
        # レイの原点を設定
        rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
        # マスクがある場合、マスク外のレイ情報をゼロに設定することも可能
        if hasattr(self, 'mask') and self.mask is not None:
            mask_expanded = np.stack([self.mask] * 3, axis=-1)  # [H, W, 3]
            rays_d = rays_d * mask_expanded  # マスク外を0ベクトルに
        pixel_coords = np.stack([i, j], axis=-1)  # [H, W, 2]
        return rays_o, rays_d, pixel_coords


    def transform_pose(self, line):
        # FLIP = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        transform_pose = np.zeros((4,4), np.float32)
        transform_pose[3,3] = 1.0
        rotation_matrix = np.array([float(x) for x in line[1:10]]).reshape(3,3)
        translation = np.array([float(x) for x in line[10:13]]).reshape(3)
        # translation[[0, 2]] = translation[[2, 0]] # xz平面で反転
        # translation[[2]] *= -1
        
        # rotation_matrix = rotation_matrix @ FLIP  
        # translation = translation 
        transform_pose[:3,:3] = rotation_matrix
        transform_pose[0:3,3] = translation.T

        return transform_pose
    def pose_trance(self,poses,scene_bounds):
        """
        ポーズとシーン境界ボックスをOpenCV座標系からOpenGL座標系に変換
        """
        # FLIP_ROTATION = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        FLIP_TRANSLATION = np.array([-1, 1, -1])
        new_poses = poses.copy()
        # for pose in new_poses:
            # translation = pose[:3, 3]
            # rotation = pose[:3, :3]
            # pose[:3, :3] = rotation @ FLIP_ROTATION
            # pose[0, 3] = scene_bounds[1]-translation[0]
            # pose[2, 3] = scene_bounds[5]-translation[2]

        new_scene_bounds = scene_bounds.copy()
        #x
        # new_scene_bounds[0] = -scene_bounds[1]
        # new_scene_bounds[1] = -scene_bounds[0]

        #Z
        # new_scene_bounds[4] = -scene_bounds[5]
        # new_scene_bounds[5] = -scene_bounds[4]
        # import pdb; pdb.set_trace()
        return new_poses,new_scene_bounds

    def interpolate_poses(self, poses, target_frames=200):
        positions = poses[:, :3, 3]     # (N, 3)
        num_original = positions.shape[0]

        interp_indices = np.linspace(0, num_original - 1, target_frames)

        interp_positions = []
        for idx in interp_indices:
            low = int(np.floor(idx))
            high = min(low + 1, num_original - 1)
            t = idx - low
            interp_pos = (1 - t) * positions[low] + t * positions[high]
            interp_positions.append(interp_pos)
        interp_positions = np.array(interp_positions)  # (target_frames, 3)

        rot = poses[0, :3, :3]

        new_poses = []
        for i in range(target_frames):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot
            pose[:3, 3] = interp_positions[i]
            new_poses.append(pose)

        return np.stack(new_poses, axis=0)  # (target_frames, 4, 4)


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def visualize_rays_on_image(image, rays_d, step=20, scale=30, output_path='rays_visualization.png'):
    """
    光線方向ベクトルを画像上に描画し、取得できていないピクセルは赤で塗りつぶす。
    
    Args:
        image (ndarray): 背景画像 [H, W, 3]
        rays_d (ndarray): 光線方向ベクトル [H, W, 3]
        step (int): ベクトル描画の間隔
        scale (float): ベクトルの長さスケーリング
        output_path (str): 出力ファイルパス
    """
    image = image.copy()
    H, W, _ = image.shape
    cx, cy = W // 2, H // 2

    # マスク：光線がゼロベクトルのピクセル
    norm = np.linalg.norm(rays_d, axis=-1)
    invalid_mask = norm < 1e-6  # しきい値は適宜調整可能

    # マスク領域を赤で塗りつぶす
    image[invalid_mask] = [255, 0, 0]  # BGR: 赤

    for v in range(0, H, step):
        for u in range(0, W, step):
            if invalid_mask[v, u]:
                continue  # 無効ピクセルはスキップ

            direction = rays_d[v, u]
            dir_unit = direction / np.linalg.norm(direction)

            start_point = (u, v)
            end_point = (
                int(u + dir_unit[0] * scale),
                int(v - dir_unit[1] * scale)
            )
            cv2.arrowedLine(image, start_point, end_point, color=(0, 255, 0), thickness=1, tipLength=0.2)


    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Ray visualization saved to: {output_path}")

if __name__ == '__main__':
    datadir = '/export/home/AokiShare/datasets/novelviewsynthesis/Original/fisyeye-LR-stable/LAB-Trim'
    
    rays_per_batch = 4096
    
    train_dataset = NeRFRayDataset(datadir, mode='train', rays_per_image=rays_per_batch)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    test_dataset = NeRFRayDataset(datadir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Train dataset virtual size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    for batch in train_loader:
        print("Training batch:")
        print("  Rays origin shape:", batch['rays_o'].shape)       # [1, rays_per_batch, 3]
        print("  Rays direction shape:", batch['rays_d'].shape)    # [1, rays_per_batch, 3]
        print("  Pixel coords shape:", batch['pixel_coords'].shape) # [1, rays_per_batch, 2]
        print("  Target RGB shape:", batch['target_rgb'].shape)    # [1, rays_per_batch, 3]
        print("  Image index:", batch['image_idx'].item())         # スカラー
        # print("  Near:", batch['near'].item()) 
        # print("  far:", batch['far'].item())         # スカラー
        
        # スカラー
        break
    
    for batch in test_loader:
        print("\nTest batch:")
        print("  Image shape:", batch['image'].shape)              # [1, H, W, 3]
        print("  Pose shape:", batch['pose'].shape)                # [1, 4, 4]
        print("  Rays origin shape:", batch['rays_o'].shape)       # [1, H, W, 3]
        print("  Rays direction shape:", batch['rays_d'].shape)    # [1, H, W, 3]
        print("  Pixel coords shape:", batch['pixel_coords'].shape) # [1, H, W, 2]
        # print("  Near:", batch['near'].item()) 
        # print("  far:", batch['far'].item())         # スカラー
         # 可視化実行
        image = batch['image'][0].numpy() * 255  # [H, W, 3], range [0, 1] → [0, 255]
        image = image.astype(np.uint8)
        rays_d = batch['rays_d'][0].numpy()      # [H, W, 3]

        visualize_rays_on_image(image, rays_d, step=30, scale=50, output_path='fisheye_ray_overlay.png')
        break

    print("\nK:", train_dataset.K)
    print("Pose normalization:", train_dataset.pose_normalization)
    print("\nRender poses shape:", train_dataset.render_poses.shape)
