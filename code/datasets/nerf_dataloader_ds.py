import os
import numpy as np
import cv2
import torch
import math
import random
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy import ndimage
from PIL import Image

class NeRFRayDataset(Dataset):
    """
    NeRF Ray Dataset with integrated DS (Double Sphere) fisheye camera support.
    
    This class combines the functionality of the original NeRFRayDataset with
    the DS (Double Sphere) fisheye camera model, providing:
    
    - Accurate DS model ray generation for fisheye cameras
    - Automatic mask generation based on DS model physical constraints  
    - Per-camera (L/R) mask handling for stereo fisheye setups
    - YAML-based camera parameter loading with automatic scaling
    - Debug mask saving and visualization capabilities
    - Enhanced pose normalization and transformation methods
    
    The DS model implementation follows the formulation from:
    "The Double Sphere Camera Model" by Usenko et al.
    """
    # def __init__(self, datadir, mode='train', rays_per_image=1024, transform=None, K=None, precache=False,img_size=(3648, 3648), precache_render_poses=False):
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
        # print(f"Camera parameters loaded from {datadir}")
        self.camera_params = self.load_camera_params_from_yaml(datadir)
        # if self.camera_params is not None:
        self.camera_params = self.scale_camera_params_to_new_resolution(
            self.camera_params, img_size[0], img_size[1]
        )
        # print(f"Camera parameters scaled to resolution: {self.W}x{self.H}")
        # print(self.camera_params)
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
        # all_poses = all_poses @ np.diag([1, 1, -1, 1]).astype(np.float32)  # OpenCVからOpenGL座標系に変換
        # print(all_poses)
        # self.normalized_poses = self.normalize_poses_translation_to_01(all_poses.copy())
        # # BBox正規化（flip_z=Trueで同じ座標系に変換）
        self.normalized_scene_bounds, self.pose_normalization = self.normalize_bbox(self._raw_scene_bounds, flip_z=False)
        self.normalized_poses = self.normalize_poses(self.raw_all_poses)
        print(self.normalized_poses)
        self.render_rays = None
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
    def load_scene_bbox(self,basedir,csv_filename: str = "scene_bbox.csv") -> dict:
        import csv
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
    def normalize_bbox(self, bbox, flip_z=True, target_range=1.0):
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
        
        # Z軸反転を考慮してcenterとscaleを計算
        if flip_z:
            # Z座標を反転
            bbox_min_transformed = bbox_min.copy()
            bbox_max_transformed = bbox_max.copy()
            bbox_min_transformed[2] *= -1.0
            bbox_max_transformed[2] *= -1.0
            # 反転後のmin/maxを再計算
            z_min = min(bbox_min_transformed[2], bbox_max_transformed[2])
            z_max = max(bbox_min_transformed[2], bbox_max_transformed[2])
            bbox_min_transformed[2] = z_min
            bbox_max_transformed[2] = z_max
        else:
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
    def normalize_poses(self, poses):
        """
        BBox由来のcenter/scaleでPoseを正規化する
        注意: 呼び出し側で事前にOpenCV→OpenGL変換が済んでいる前提
        """
        assert hasattr(self, "center") and hasattr(self, "scale"), \
            "Call normalize_bbox() before normalize_poses()"

        poses_norm = poses.copy()
        poses_norm[:, :3, 3] = (poses[:, :3, 3] - self.center) / self.scale

        # flip_z変換は既に呼び出し側で実行済みなので、ここでは行わない
        # （normalize_bboxでZ軸反転されたcenter/scaleを使用するだけ）

        print("[Pose Normalization] applied from BBox center/scale.")
        return poses_norm 
    def _generate_ds_valid_mask(self, cam_params, width, height, max_radius_ratio=0.48):
        """DSパラメータから有効画素maskを生成する（物理的投影可能領域のみ）。"""
        fx = cam_params['fx']
        fy = cam_params['fy']
        cx = cam_params['cx']
        cy = cam_params['cy']
        alpha = cam_params['alpha']
        xi = cam_params['xi']
        
        # 全画素座標から正規化座標の2乗(r2)を計算
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        xn = (xx - cx) / fx
        yn = (yy - cy) / fy
        r2 = xn**2 + yn**2

        # 1. DSモデルの物理的制約に基づくマスク（緩和版）
        # まず実用的な半径制限を優先
        distance_from_center = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        max_radius = min(width, height) * max_radius_ratio
        distance_mask = distance_from_center < max_radius
        
        # 2. DSモデルの理論的制約をチェック（ただし緩和）
        if alpha <= 0.5:
            # alphaが0.5以下なら、r2に関わらず常に非負になるため、全域を有効とする
            physical_mask = np.ones_like(r2, dtype=bool)
        else:
            # alphaが0.5より大きい場合、r2には上限が存在するが、緩和して使用
            max_r2 = 1.0 / (2 * alpha - 1)
            # 理論値の1.2倍まで許可（実用性を重視）
            physical_mask = (r2 <= max_r2 * 1.2)
        
        # 3. マスクを結合（実用的な半径制限を優先）
        combined_mask = distance_mask  # まず距離マスクをベースに
        combined_mask = combined_mask & physical_mask  # 物理的制約も考慮
        
        # 4. (任意) マスクの穴埋めなどの後処理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        
        # 5. (任意) 最大の連結成分のみを保持
        labeled_mask, num_labels = ndimage.label(combined_mask)
        if num_labels > 1:
            label_sizes = [(labeled_mask == i).sum() for i in range(1, num_labels + 1)]
            largest_label = np.argmax(label_sizes) + 1
            combined_mask = (labeled_mask == largest_label)
            
        print(f"DS Mask Generation - Camera params: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  alpha={alpha:.4f}, xi={xi:.4f}")
        print(f"  Distance mask: {np.sum(distance_mask)}/{distance_mask.size} ({100*np.sum(distance_mask)/distance_mask.size:.1f}%)")
        print(f"  Physical mask: {np.sum(physical_mask)}/{physical_mask.size} ({100*np.sum(physical_mask)/physical_mask.size:.1f}%)")
        print(f"  Combined mask: {np.sum(combined_mask)}/{combined_mask.size} ({100*np.sum(combined_mask)/combined_mask.size:.1f}%)")
        print(f"  Max radius used: {max_radius:.1f} pixels (ratio: {max_radius_ratio})")
        print(f"  Mask center: ({cx:.1f}, {cy:.1f})")
        return combined_mask
    
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
            if 'L' in img_path:
                camera_params = self.camera_params['L']
            else:
                camera_params = self.camera_params['R']
                
            rays_o, rays_d, pixel_coords = self.get_rays_fisheye_ds(self.H, self.W, 
                                                                    camera_params["fx"], camera_params["fy"],
                                                                    camera_params["cx"], camera_params["cy"],
                                                                    camera_params["alpha"], camera_params["xi"],
                                                                    torch.from_numpy(pose.copy()).float()
                                                                    )
            
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
    @property
    def total_valid_rays(self):
        """総Rayの数を返す（マスク関係なし）"""
        return self.H * self.W * len(self.train_img_paths)
    
    @property
    def valid_pixels_per_image(self):
        """1画像あたりの総ピクセル数を返す（マスク関係なし）"""
        return self.H * self.W
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
                if 'L' in img_path:
                    camera_params = self.camera_params['L']
                else:
                    camera_params = self.camera_params['R']
                    
                rays_o, rays_d, pixel_coords = self.get_rays_fisheye_ds(self.H, self.W, 
                                                                        camera_params["fx"], camera_params["fy"],
                                                                        camera_params["cx"], camera_params["cy"],
                                                                        camera_params["alpha"], camera_params["xi"],
                                                                        torch.from_numpy(pose.copy()).float()
                                                                        )
                # mask_tensor = torch.tensor(self.mask, dtype=torch.bool, device=rays_o.device)
                
                # near, far = self.compute_near_far_from_aabb(
                #     rays_o, 
                #     rays_d, 
                #     self.scene_bounds, 
                # mask_tensor)
            # DSモデル用のマスクを考慮してサンプリング
            h, w = img.shape[:2]
            
            # カメヤキーを取得
            cam_key = None
            if hasattr(self, 'image_camkeys') and img_idx < len(self.image_camkeys):
                cam_key = self.image_camkeys[img_idx]
            elif 'L' in self.img_paths[img_idx]:
                cam_key = 'L'
            elif 'R' in self.img_paths[img_idx]:
                cam_key = 'R'
            else:
                cam_key = 'L'  # デフォルト
            
            # DSマスクを使用（ある場合）
            current_mask = None
            if hasattr(self, 'masks') and cam_key in self.masks:
                current_mask = self.masks[cam_key]
            elif self.mask is not None:
                current_mask = self.mask
                
            if current_mask is not None:
                # マスク内のピクセルからのみサンプリング
                valid_indices = np.where(current_mask.flatten())[0]
                if len(valid_indices) < self.rays_per_image:
                    # 有効なピクセルが少ない場合は重複を許可
                    select_flat_idxs = np.random.choice(valid_indices, size=self.rays_per_image, replace=True)
                else:
                    # 十分な有効ピクセルがある場合は重複なし
                    select_flat_idxs = np.random.choice(valid_indices, size=self.rays_per_image, replace=False)
                
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


            # サンプルのバッチを返す（実際に使用されたマスクを返す）
            sample = {
                'pose' : torch.from_numpy(self.poses[img_idx].copy()).float(),  # [4, 4]
                'rays_o': rays_o_sampled,                   # [rays_per_image, 3]
                'rays_d': rays_d_sampled,                   # [rays_per_image, 3]
                'pixel_coords': pixel_coords_sampled,       # [rays_per_image, 2]
                'target_rgb': target_rgb,                   # [rays_per_image, 3]
                'image_idx': torch.tensor(img_idx, dtype=torch.int32),  # 画像インデックス
                'mask': torch.from_numpy(current_mask).bool() if current_mask is not None else None,  # [H, W] - 実際に使用されたマスク
                'cam_key': cam_key,  # デバッグ用にカメラキーも追加
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
                
                # DSマスク適用（カメラ毎）
                cam_key = None
                if hasattr(self, 'image_camkeys') and img_idx < len(self.image_camkeys):
                    cam_key = self.image_camkeys[img_idx]
                elif 'L' in img_path:
                    cam_key = 'L'
                elif 'R' in img_path:
                    cam_key = 'R'
                else:
                    cam_key = 'L'  # デフォルト
                    
                current_mask = None
                if hasattr(self, 'masks') and cam_key in self.masks:
                    current_mask = self.masks[cam_key]
                elif self.mask is not None:
                    current_mask = self.mask
                    
                if current_mask is not None:
                    mask_3ch = np.stack([current_mask] * 3, axis=-1)  # 3チャンネルに拡張
                    img = img * mask_3ch  # マスク外を0（黒）に
                
                # レイ計算
                if 'L' in img_path:
                    camera_params = self.camera_params['L']
                else:
                    camera_params = self.camera_params['R']
                    
                mask_tensor = torch.tensor(self.mask, dtype=torch.bool) if self.mask is not None else None
                rays_o, rays_d, pixel_coords = self.get_rays_fisheye_ds(self.H, self.W, 
                                                                        camera_params["fx"], camera_params["fy"],
                                                                        camera_params["cx"], camera_params["cy"],
                                                                        camera_params["alpha"], camera_params["xi"],
                                                                        torch.from_numpy(pose.copy()).float()
                                                                        )
                # near, far = self.compute_near_far_from_aabb(rays_o, rays_d, self.scene_bounds, mask=mask_tensor)
                # サンプルを返す（現在使用されているマスクを返す）
                sample = {
                    'image': torch.from_numpy(img.copy()).float(),           # [H, W, 3]
                    'pose': torch.from_numpy(pose.copy()).float(),           # [4, 4]
                    'rays_o': torch.from_numpy(rays_o.copy()).float(),       # [H, W, 3]
                    'rays_d': torch.from_numpy(rays_d.copy()).float(),       # [H, W, 3]
                    'pixel_coords': torch.from_numpy(pixel_coords.copy()).float(),  # [H, W, 2]
                    'mask': torch.from_numpy(current_mask).bool() if current_mask is not None else None,  # [H, W] - 実際に使用されたマスク
                    'cam_key': cam_key,  # デバッグ用にカメラキーも追加
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
        # 元の解像度を一度だけ保存（初回読み込み時）
        # if self.W == 0 and self.H == 0:
        #     self.gt_w, self.gt_h = img.shape[1], img.shape[0]
        #     print(f"Original image resolution: {self.gt_w}x{self.gt_h}")
        #     print(f"Target resolution: {self.img_size[0]}x{self.img_size[1]}")
        img = cv2.resize(img, self.img_size)  # 指定サイズにリサイズ
        print("Loaded image:", img_path, "->", img.shape)
        return img.astype(np.float32)

    def load_camera_params_from_yaml(self, data_dir):
        """
        Load Double Sphere camera parameters from YAML files.
        Expects left_camera-camchain.yaml and right_camera-camchain.yaml files.

        Parameters:
        - data_dir (str): Path to the directory containing YAML files

        Returns:
        - dict: Dictionary like {'L': {...}, 'R': {...}} where each is a dict of camera parameters
        """
        params = {}
        
        # Load left camera parameters
        left_yaml_path = os.path.join(data_dir, 'left_camera-camchain.yaml')
        if os.path.exists(left_yaml_path):
            with open(left_yaml_path, 'r') as f:
                left_data = yaml.safe_load(f)
                cam_data = left_data['cam0']
                intrinsics = cam_data['intrinsics']
                resolution = cam_data['resolution']  # [width, height]
                # DSクラスと同じ順序: [xi, alpha, fx, fy, cx, cy]
                params['L'] = {
                    'xi': intrinsics[0],
                    'alpha': intrinsics[1],
                    'fx': intrinsics[2],
                    'fy': intrinsics[3],
                    'cx': intrinsics[4],
                    'cy': intrinsics[5],
                    'original_width': resolution[0],
                    'original_height': resolution[1]
                }
                # print(f"Loaded left camera params: {params['L']}")
        else:
            print(f"Warning: Left camera YAML not found at {left_yaml_path}")
        
        # Load right camera parameters
        right_yaml_path = os.path.join(data_dir, 'right_camera-camchain.yaml')
        if os.path.exists(right_yaml_path):
            with open(right_yaml_path, 'r') as f:
                right_data = yaml.safe_load(f)
                cam_data = right_data['cam0']
                intrinsics = cam_data['intrinsics']
                resolution = cam_data['resolution']  # [width, height]
                # DSクラスと同じ順序: [xi, alpha, fx, fy, cx, cy]
                params['R'] = {
                    'xi': intrinsics[0],
                    'alpha': intrinsics[1],
                    'fx': intrinsics[2], 
                    'fy': intrinsics[3],
                    'cx': intrinsics[4],
                    'cy': intrinsics[5],
                    'original_width': resolution[0],
                    'original_height': resolution[1]
                }
                # print(f"Loaded right camera params: {params['R']}")
        else:
            print(f"Warning: Right camera YAML not found at {right_yaml_path}")
        
        if not params:
            print("Warning: No camera parameters loaded. Using default values.")
            return None
        
        return params

    def scale_camera_params_to_new_resolution(self, camera_params, new_width, new_height):
        """
        カメラパラメータを新しい解像度に合わせてスケーリング

        Parameters:
        - camera_params (dict): 元のカメラパラメータ
        - new_width (int): 新しい画像の幅
        - new_height (int): 新しい画像の高さ

        Returns:
        - dict: スケーリングされたカメラパラメータ
        """
        if camera_params is None:
            return None
            
        scaled_params = {}
        
        for cam_key, params in camera_params.items():
            orig_width = params['original_width']
            orig_height = params['original_height']
            
            # スケーリング係数を計算
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height
            
            scaled_params[cam_key] = {
                'alpha': params['alpha'],  # alphaとxiは解像度に依存しない
                'xi': params['xi'],
                'fx': params['fx'] * scale_x,  # 焦点距離をスケーリング
                'fy': params['fy'] * scale_y,
                'cx': params['cx'] * scale_x,  # 主点をスケーリング
                'cy': params['cy'] * scale_y,
                'original_width': orig_width,
                'original_height': orig_height,
                'scaled_width': new_width,
                'scaled_height': new_height,
                'scale_x': scale_x,
                'scale_y': scale_y
            }
            
            print(f"Scaled {cam_key} camera params from ({orig_width}x{orig_height}) to ({new_width}x{new_height})")
            print(f"  Scale factors: x={scale_x:.4f}, y={scale_y:.4f}")
            print(f"  fx: {params['fx']:.2f} -> {scaled_params[cam_key]['fx']:.2f}")
            print(f"  fy: {params['fy']:.2f} -> {scaled_params[cam_key]['fy']:.2f}")
            print(f"  cx: {params['cx']:.2f} -> {scaled_params[cam_key]['cx']:.2f}")
            print(f"  cy: {params['cy']:.2f} -> {scaled_params[cam_key]['cy']:.2f}")
        
        return scaled_params

    # def _get_rays_ds(self, pix_x, pix_y, fx, fy, cx, cy, alpha, xi, c2w):
    #     """DSモデルによる光線生成"""
    #     # 正規化座標
    #     x = (pix_x - cx) / fx
    #     y = -(pix_y - cy) / fy  # 上向きを正に
    #     # DSモデルの計算
    #     d1 = np.sqrt(x**2 + y**2 + 1)
    #     d2 = np.sqrt(x**2 + y**2 + (1 - xi * d1)**2)

    #     denom = alpha * d2 + (1 - alpha) * (xi * d1 + 1)
    #     X = x / denom
    #     Y = y / denom
    #     Z = (xi * d1 + 1) / denom

    #     dirs = np.stack([X, Y, -Z], axis=-1)
    #     dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-8)
    #     # ---DSモデルによる方向ベクトル計算終了---
    #     # ワールド座標に変換
    #     dirs2 = dirs.reshape(-1, 3)
    #     world = np.matmul(dirs2, c2w[:3, :3].T).reshape(dirs.shape)
    #     origins = np.broadcast_to(c2w[:3, 3], world.shape)
    #     return origins, world
    
    def _get_rays_ds(self, pix_x, pix_y, fx, fy, cx, cy, alpha, xi, c2w):
        """DSモデルによる正確な光線生成（修正版）"""
        # xi = 0.0 # DSモデルをUCMに簡略化
        # 1. 正規化座標への変換
        mx = (pix_x - cx) / fx
        my = -(pix_y - cy) / fy  # 上向きを正とするユーザーの実装に合わせます
        r2 = mx**2 + my**2

        # 2. DSモデルの正しい逆変換 (Unprojection)
        # 公式の逆変換導出に基づく計算
        
        # アルファ由来の項の計算（ルートの中が負にならないようクリップ推奨）
        # term1 = 1 - (2*alpha - 1) * r2
        term1 = np.maximum(1 - (2 * alpha - 1) * r2, 0)
        
        # 中間変数 mz の計算
        # mz = (1 - alpha^2 * r2) / (alpha * sqrt(term1) + 1 - alpha)
        mz = (1 - alpha**2 * r2) / (alpha * np.sqrt(term1) + (1 - alpha))
        
        # xi (オフセット) を考慮した係数の計算
        # term2 = mz^2 + (1 - xi^2) * r2
        term2 = np.maximum(mz**2 + (1 - xi**2) * r2, 0)
        k = (mz * xi + np.sqrt(term2)) / (mz**2 + r2)
        
        # 3. 単位球上の座標 (Ray方向) を算出
        X = k * mx
        Y = k * my
        Z = k * mz - xi

        # スタックして正規化
        dirs = np.stack([X, Y, -Z], axis=-1)
        # 理論上は既に正規化に近い値ですが、数値誤差除去のため正規化
        dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-10)

        # 4. ワールド座標に変換
        dirs2 = dirs.reshape(-1, 3)
        world = np.matmul(dirs2, c2w[:3, :3].T).reshape(dirs.shape)
        origins = np.broadcast_to(c2w[:3, 3], world.shape)
        return origins, world

    def get_rays_fisheye_ds(self, H, W, fx, fy, cx, cy, alpha, xi, c2w):
        """
        Fisheyeレンズ（Double Sphereモデル）に基づく光線生成（改良版）

        Args:
            H, W: 画像の高さと幅
            fx, fy: 焦点距離（ピクセル単位）
            cx, cy: 主点（画像中心）
            alpha, xi: DSモデルのパラメータ
            c2w: カメラ→ワールド変換行列（3x4または4x4）

        Returns:
            rays_o: (H, W, 3) レイの原点（world座標）
            rays_d: (H, W, 3) レイの方向（正規化済, world座標）
            pixel_coords: (H, W, 2) ピクセル座標

        """
        
        device = c2w.device
        
        
        # pixel grid
        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=device),
            torch.arange(H, dtype=torch.float32, device=device),
            indexing="xy",
        )

        # numpy版のDSモデルを使用してより正確な計算を行う
        c2w_np = c2w.cpu().numpy()
        pix_x_np = i.cpu().numpy()
        pix_y_np = j.cpu().numpy()
        
        rays_o_np, rays_d_np = self._get_rays_ds(
            pix_x_np, pix_y_np, fx, fy, cx, cy, alpha, xi, c2w_np
        )
        
        pixel_coords = torch.stack([i, j], dim=-1)

        # numpyに変換（必要なら）
        return rays_o_np, rays_d_np, pixel_coords.cpu().numpy()

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
            fisheye_fov_rad = np.pi / 2  # 180° fisheyeならπ/2

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
    def roll_rotation_matrix(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        return np.array([
            [1, 0, 0],
            [0, math.cos(angle_radians), -math.sin(angle_radians)],
            [0, math.sin(angle_radians), math.cos(angle_radians)]
        ])
    def _transform_pose(self, line):
        """DSクラス版のポーズ変換メソッド。"""
        transform_pose = np.zeros((4, 4), dtype=np.float32)
        transform_pose[3, 3] = 1.0
        
        rotation_matrix = np.array([float(x) for x in line[1:10]]).reshape(3, 3)
        translation = np.array([float(x) for x in line[10:13]])
        
        transform_pose[:3, :3] = rotation_matrix
        transform_pose[:3, 3] = translation
        
        return transform_pose
        
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

    def normalize(self, x):
        return x / np.linalg.norm(x)

    def viewmatrix(self, z, up, pos):
        vec2 = self.normalize(z)
        vec1_avg = up
        vec0 = self.normalize(np.cross(vec1_avg, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def poses_avg(self, poses):
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self.normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self.viewmatrix(vec2, up, center), hwf], 1)
        return c2w

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
    def _normalize_poses_translation_to_pi(self, poses):
        """DSクラス版のポーズ正規化メソッド。"""
        poses = poses.copy()
        translations = poses[:, :3, 3]
        
        # Center translations
        center = np.mean(translations, axis=0, keepdims=True)
        centered_trans = translations - center
        
        # Scale to pi range
        bbox_size = np.max(np.abs(centered_trans), axis=0, keepdims=True)
        max_scale = np.max(bbox_size)
        scale_factor = np.pi / (max_scale + 1e-8)
        
        normalized_trans = centered_trans * scale_factor
        
        for i in range(poses.shape[0]):
            poses[i, :3, 3] = normalized_trans[i]
        
        return poses
        
    def normalize_poses_translation_to_pi(self, poses):
        """
        全体の重心を原点に移動し、最大値がπになるようにスケーリング。
        XYZ軸の相対比率はそのまま保持。
        """

        poses = poses.copy()
        translations = poses[:, :3, 3]  # [N, 3]

        # --- 重心を原点に移動 ---
        center = np.mean(translations, axis=0, keepdims=True)
        centered_trans = translations - center

        # --- 最大軸幅に合わせたスケール係数計算 ---
        bbox_size = np.max(np.abs(centered_trans), axis=0, keepdims=True)  # [1, 3]
        max_scale = np.max(bbox_size)  # 最も大きい軸の幅

        # --- πにスケール ---
        scale_factor = np.pi / (max_scale + 1e-8)  # 0除算回避

        # --- スケーリング ---
        normalized_trans = centered_trans * scale_factor  # 比率を保持したまま π に収める

        # --- 回転行列はそのまま、平行移動だけ更新 ---
        for i in range(poses.shape[0]):
            poses[i, :3, 3] = normalized_trans[i]

        # --- スケール情報を保存（元に戻すとき用） ---
        self.pose_normalization = {
            "center": center.squeeze(),
            "scale": float(1.0 / scale_factor)  # 元に戻すとき用
        }
        # import pdb ; pdb.set_trace()
        return poses
        
    def precompute_render_pose_rays(self):
        """レンダリングポーズ（動画生成用）のレイ情報を事前計算する"""
        self.render_rays = []
        
        print(f"Precomputing rays for {len(self.render_poses)} render poses...")
        
        for i, pose in enumerate(tqdm(self.render_poses)):
            # デフォルトのカメラパラメータを使用（または最初のカメラ）
            if self.camera_params is not None:
                camera_params = list(self.camera_params.values())[0]  # 最初のカメラを使用
                rays_o, rays_d, pixel_coords = self.get_rays_fisheye_ds(
                    self.H, self.W,
                    camera_params["fx"], camera_params["fy"],
                    camera_params["cx"], camera_params["cy"], 
                    camera_params["alpha"], camera_params["xi"],
                    torch.from_numpy(pose.copy()).float()
                )
            else:
                # フォールバック（通常のfisheye計算）
                rays_o, rays_d, pixel_coords = self.get_rays_np_fisyeye(self.H, self.W, self.K, pose)
            
            self.render_rays.append({
                'rays_o': torch.from_numpy(rays_o.copy()).float(),
                'rays_d': torch.from_numpy(rays_d.copy()).float(), 
                'pixel_coords': torch.from_numpy(pixel_coords.copy()).float()
            })
        
        print(f"Completed precomputing render rays.")

    def get_render_rays(self, idx):
        """
        レンダリングポーズのレイ情報を取得
        
        Args:
            idx (int): レンダリングポーズのインデックス
            
        Returns:
            dict: レイ情報を含む辞書
        """
        if self.render_rays is None:
            # レンダリングポーズのレイをオンザフライで計算
            pose = self.render_poses[idx]
            if self.camera_params is not None:
                camera_params = list(self.camera_params.values())[0]  # 最初のカメラを使用
                rays_o, rays_d, pixel_coords = self.get_rays_fisheye_ds(
                    self.H, self.W,
                    camera_params["fx"], camera_params["fy"], 
                    camera_params["cx"], camera_params["cy"],
                    camera_params["alpha"], camera_params["xi"],
                    torch.from_numpy(pose.copy()).float()
                )
            else:
                rays_o, rays_d, pixel_coords = self.get_rays_np_fisyeye(self.H, self.W, self.K, pose)
            
            return {
                'rays_o': torch.from_numpy(rays_o.copy()).float(),
                'rays_d': torch.from_numpy(rays_d.copy()).float(),
                'pixel_coords': torch.from_numpy(pixel_coords.copy()).float(),
                'c2w': torch.from_numpy(pose.copy()).float()
            }
        else:
            return self.render_rays[idx]

    def save_masks(self, outdir: str = None):
        """
        per-camera マスクをファイル保存する。
        保存先: <datadir>/debug_masks/ もしくは outdir で指定
        出力: mask_L.png, mask_R.png, おまけで重ね合わせの可視化 composite.png
        リサイズファクターも考慮して適切なサイズでマスクを保存する。
        """
        if not hasattr(self, 'masks') or not self.masks:
            print("No masks to save.")
            return
            
        if outdir is None:
            outdir = os.path.join(self.datadir, 'debug_masks')
        os.makedirs(outdir, exist_ok=True)

        # 個別保存
        saved = []
        for key in self.masks.keys():  # 'L','R'
            m = (self.masks[key] * 255).astype(np.uint8)
            path = os.path.join(outdir, f"mask_{key}.png")
            Image.fromarray(m).save(path)
            saved.append(path)
            
            # リサイズ情報をファイル名に含めた追加保存
            if hasattr(self, 'camera_params') and self.camera_params and key in self.camera_params:
                cam_params = self.camera_params[key]
                if 'scale_x' in cam_params and 'scale_y' in cam_params:
                    scale_info = f"_scaled_{cam_params['scale_x']:.2f}x{cam_params['scale_y']:.2f}"
                    scaled_path = os.path.join(outdir, f"mask_{key}{scale_info}.png")
                    Image.fromarray(m).save(scaled_path)
                    saved.append(scaled_path)

        # （任意）L/RをRGBに重ねた可視化も作る
        if 'L' in self.masks and 'R' in self.masks:
            H, W = self.masks['L'].shape
            comp = np.zeros((H, W, 3), dtype=np.uint8)
            comp[..., 0] = (self.masks['L'] * 255).astype(np.uint8)  # R: L
            comp[..., 1] = (self.masks['R'] * 255).astype(np.uint8)  # G: R
            # Bチャンネルは L∧R の共通部にしてもOK（視覚的にわかりやすい）
            comp[..., 2] = ((self.masks['L'] & self.masks['R']) * 255).astype(np.uint8)
            comp_path = os.path.join(outdir, "mask_composite_LRG.png")
            Image.fromarray(comp).save(comp_path)
            saved.append(comp_path)
            
            # スケール情報付きの複合マスクも保存
            if (hasattr(self, 'camera_params') and self.camera_params and 
                'L' in self.camera_params and 'scale_x' in self.camera_params['L']):
                scale_x = self.camera_params['L']['scale_x']
                scale_y = self.camera_params['L']['scale_y'] 
                scaled_comp_path = os.path.join(outdir, f"mask_composite_LRG_scaled_{scale_x:.2f}x{scale_y:.2f}.png")
                Image.fromarray(comp).save(scaled_comp_path)
                saved.append(scaled_comp_path)

        # マスク情報をテキストファイルに保存
        info_path = os.path.join(outdir, "mask_info.txt")
        with open(info_path, 'w') as f:
            f.write("DS Mask Information\n")
            f.write("=" * 30 + "\n")
            f.write(f"Dataset: {self.datadir}\n")
            f.write(f"Final image size: {self.W} x {self.H}\n")
            
            for key in self.masks.keys():
                f.write(f"\n{key} Camera:\n")
                if hasattr(self, 'camera_params') and self.camera_params and key in self.camera_params:
                    cam_params = self.camera_params[key]
                    
                    # 元の解像度情報
                    if 'original_width' in cam_params:
                        f.write(f"  Original size: {cam_params['original_width']} x {cam_params['original_height']}\n")
                    
                    # スケール情報
                    if 'scale_x' in cam_params:
                        f.write(f"  Scale factors: {cam_params['scale_x']:.3f} x {cam_params['scale_y']:.3f}\n")
                        
                    # カメラパラメータ（スケール済み）
                    f.write(f"  Scaled parameters:\n")
                    f.write(f"    fx={cam_params['fx']:.2f}, fy={cam_params['fy']:.2f}\n")
                    f.write(f"    cx={cam_params['cx']:.2f}, cy={cam_params['cy']:.2f}\n")
                    f.write(f"    alpha={cam_params['alpha']:.4f}, xi={cam_params['xi']:.4f}\n")
                    
                # マスクのカバレッジ
                mask_coverage = np.sum(self.masks[key]) / self.masks[key].size
                f.write(f"  Mask coverage: {mask_coverage*100:.1f}% ({np.sum(self.masks[key])}/{self.masks[key].size} pixels)\n")
        
        saved.append(info_path)
        print("[Mask] Saved:", *saved, sep="\n  ")
        
    def apply_mask_to_render(self, rgb_image: np.ndarray, cam_idx: int) -> np.ndarray:
        """
        推論レンダリング結果に per-camera マスクを掛けて無効領域を黒にする。
        Args:
            rgb_image: (H, W, 3) float32 in [0,1]
            cam_idx:   int（このフレームのカメラIDX）
        Returns:
            (H, W, 3) float32 in [0,1]
        """
        if not hasattr(self, 'masks') or not hasattr(self, 'image_camkeys'):
            return rgb_image
            
        if cam_idx >= len(self.image_camkeys):
            return rgb_image
            
        cam_key = self.image_camkeys[cam_idx]  # 'L' or 'R'
        if cam_key not in self.masks:
            return rgb_image
            
        mask = self.masks[cam_key]             # (H, W) bool
        return rgb_image * mask[..., None].astype(np.float32)
        
    def debug_mask_info(self, img_idx=0):
        """
        デバッグ用：指定した画像インデックスのマスク情報を詳細に出力
        """
        print(f"\n=== Debug Mask Info for image {img_idx} ===")
        
        if img_idx >= len(self.img_paths):
            print(f"Error: img_idx {img_idx} is out of range (max: {len(self.img_paths)-1})")
            return
            
        img_path = self.img_paths[img_idx]
        print(f"Image path: {img_path}")
        
        # カメラキーの確認
        cam_key = None
        if hasattr(self, 'image_camkeys') and img_idx < len(self.image_camkeys):
            cam_key = self.image_camkeys[img_idx]
        elif 'L' in img_path:
            cam_key = 'L'
        elif 'R' in img_path:
            cam_key = 'R'
        else:
            cam_key = 'L'
            
        print(f"Camera key: {cam_key}")
        
        # マスクの確認
        print(f"Available masks: {list(self.masks.keys()) if hasattr(self, 'masks') else 'None'}")
        print(f"DS auto mask enabled: {getattr(self, 'ds_auto_mask', 'Unknown')}")
        
        if hasattr(self, 'masks') and cam_key in self.masks:
            mask = self.masks[cam_key]
            print(f"DS mask shape: {mask.shape}")
            print(f"DS mask coverage: {np.sum(mask)}/{mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
            print(f"DS mask type: {type(mask)}, dtype: {mask.dtype}")
        else:
            print("No DS mask available for this camera")
            
        if hasattr(self, 'mask') and self.mask is not None:
            print(f"Legacy mask shape: {self.mask.shape}")
            print(f"Legacy mask coverage: {np.sum(self.mask)}/{self.mask.size} ({100*np.sum(self.mask)/self.mask.size:.1f}%)")
        else:
            print("No legacy mask available")
            
        # カメラパラメータの確認
        if hasattr(self, 'camera_params') and self.camera_params and cam_key in self.camera_params:
            cam_params = self.camera_params[cam_key]
            print(f"Camera params for {cam_key}:")
            print(f"  fx={cam_params['fx']:.2f}, fy={cam_params['fy']:.2f}")
            print(f"  cx={cam_params['cx']:.2f}, cy={cam_params['cy']:.2f}")
            print(f"  alpha={cam_params['alpha']:.4f}, xi={cam_params['xi']:.4f}")
            if 'scale_x' in cam_params:
                print(f"  Scale: {cam_params['scale_x']:.3f}x{cam_params['scale_y']:.3f}")
        else:
            print(f"No camera params for {cam_key}")
            
        print("=" * 50)
        
    # def normalize_poses_translation_to_01(self, poses):
    #     """
    #     全体の重心を原点に移動し、最大半径でスケーリングして [-1, 1] に正規化。
    #     XYZ軸の相対関係を壊さない方法。
    #     """

    #     poses = poses.copy()
    #     import pdb ; pdb.set_trace()
    #     translations = poses[:, :3, 3]  # [N, 3]

    #     # --- 平行移動の正規化 ---
    #     center = np.mean(translations, axis=0, keepdims=True)
    #     centered_trans = translations - center
    #     max_radius = np.pi /np.max(np.linalg.norm(centered_trans, axis=1, keepdims=True))
    #     normalized_trans = centered_trans / (max_radius + 1e-8)

    #     # --- 回転行列の再定義（中心から見た方向をforwardとする） ---
    #     for i in range(poses.shape[0]):
    #         cam_pos = normalized_trans[i]  # 正規化後の位置
    #         # 一旦回転は無視#
    #         # forward = cam_pos / (np.linalg.norm(cam_pos) + 1e-8)  # 原点→カメラ位置方向

    #         # tmp_up = np.array([0, 1, 0], dtype=np.float32)
    #         # if np.abs(np.dot(forward, tmp_up)) > 0.99:
    #         #     tmp_up = np.array([0, 0, 1], dtype=np.float32)

    #         # right = np.cross(tmp_up, forward)
    #         # right = right / (np.linalg.norm(right) + 1e-8)

    #         # up = np.cross(forward, right)

    #         # R = np.stack([right, up, forward], axis=1)  # [3, 3]
    #         # poses[i, :3, :3] = R
    #         poses[i, :3, 3] = cam_pos  # 更新後の位置も入れる
    #     import pdb; pdb.set_trace()

    #     self.pose_normalization = {
    #         "center": center.squeeze(),
    #         "scale": float(max_radius)
    #     }

    #     return poses

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
    datadir = '/export/home/AokiShare/datasets/novelviewsynthesis/Original/fisyeye-LR-stable/makudomae-fisheye-nerf-axixs'
    
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
        # print("  camera_params:", batch['K'])              # [1, H, W, 3]
        break
    
    for batch in test_loader:
        print("\nTest batch:")
        print("  Image shape:", batch['image'].shape)              # [1, H, W, 3]
        print("  Pose shape:", batch['pose'].shape)                # [1, 4, 4]
        print("  Rays origin shape:", batch['rays_o'].shape)       # [1, H, W, 3]
        print("  Rays direction shape:", batch['rays_d'].shape)    # [1, H, W, 3]
        print("  Pixel coords shape:", batch['pixel_coords'].shape) # [1, H, W, 2]
        # print("  camera_params:", batch['K'])              # [1, H, W, 3]
        break
    
    print("\nRender poses shape:", train_dataset.render_poses.shape)
    print("Camera parameters (scaled):", train_dataset.camera_params)
    if train_dataset.camera_params:
        for cam_key, params in train_dataset.camera_params.items():
            if 'scale_x' in params:
                print(f"{cam_key} camera - Original: {params['original_width']}x{params['original_height']}, "
                      f"Scaled: {params['scaled_width']}x{params['scaled_height']}, "
                      f"Scale factors: {params['scale_x']:.3f}x{params['scale_y']:.3f}")
