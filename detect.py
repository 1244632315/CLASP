import os
import cv2
import sep
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from astropy.io import fits
from datetime import datetime
import imageio.v2 as imageio

from tool import load_img, show_img, draw3D, ax_imshow, get_fileList, get_roi, convert_16_to_RGB
from extract import extract, extract_sep, extract_NTH, nms
from register import TriAngleRectifyWithDelaunay, registerPoints
from kalman import kalman_smooth


class PT(object):
    def __init__(self, idx, x, y, vx, vy, status, conf, x0=-1, y0=-1):
        self.idx    = idx
        self.x      = x
        self.y      = y
        self.x0     = x if x0 == -1 else x0
        self.y0     = y if y0 == -1 else y0
        self.vx     = vx
        self.vy     = vy
        self.status = status
        self.conf   = conf


class Track(object):
    def __init__(self, pt1, pt2, H0):
        self.len = 2
        self.idx0 = pt1.idx
        self.cur_idx = pt2.idx
        self.register_idx = pt2.idx
        self.cur_pt = pt2
        self.traj = [pt1, pt2]
        self.update_speed()
        self.conf = 1
        self.Hs = [H0]

    def append(self, pt: PT):
        self.traj.append(pt)
        # update
        self.cur_idx = pt.idx
        self.register_idx = pt.idx
        self.len += 1
        self.cur_pt = pt
        self.update_speed()
        self.update_conf()

    def get(self, key=None, idx=None):
        oup = np.array([getattr(x, key) for x in self.traj])
        if idx is None:
            return oup
        else:
            return oup[idx]

    def update(self, H12):
        for i, pt in enumerate(self.traj):
            (xr, yr) = registerPoints((pt.x, pt.y), H12)
            pt.x, pt.y = xr, yr
        for i, Hi in enumerate(self.Hs):
            self.Hs[i] = np.dot(H12, Hi)
        self.Hs.append(H12)
    
    def update_speed(self, ):
        vxs = self.get('vx')
        vys = self.get('vy')
        self.vx = np.mean(vxs)
        self.vy = np.mean(vys)
    
    def update_conf(self, ):
        confs = self.get('status')
        self.conf = sum(confs) / self.len
    
    def print(self, ):
        con = ''
        for pt in self.traj:
            con += f'\tframe-{pt.idx:02d}  status:{pt.status}  conf:{pt.conf*100:02f}%  pt:({pt.x:6.2f},{pt.y:6.2f})  '
            con += f'pt0:({pt.x0:6.2f},{pt.y0:6.2f})  v:({pt.vx:5.2f}, {pt.vy:5.2f})\n'
        print(con)
    
    def show(self, fig=None):
        if fig is None:
            plt.figure()
        xs, ys = self.get('x'), self.get('y')
        lab = f'{self.idx0} - {self.cur_idx}'
        plt.plot(xs, ys, ms=2, label=lab, marker='o', linestyle='-')

    def remove_invalid_points(self, ):
        # delete final invalid predictions
        for i, pt in enumerate(self.traj[::-1]):
            if pt.status == 0:
                self.traj.pop()
            else:
                break
        self.update_speed()
        self.len = len(self.traj)

    def interpolate(self, ):
        xs, ys, status = self.get('x'), self.get('y'), self.get('status')
        data = np.stack([status, xs, ys]).reshape(3, self.len).T
        new_data = kalman_smooth(data)
        new_traj = []
        # interpolate leak detections
        for i, pt in enumerate(self.traj[:-1]):
            if pt.status == 0:
                pt.x, pt.y = new_data[i, 1], new_data[i, 2]
                pt.x0, pt.y0 = registerPoints((pt.x, pt.y), np.linalg.inv(self.Hs[i]))
            new_traj.append(pt)
        new_traj.append(self.traj[-1])
        self.traj = new_traj


class DBT(object):
    def __init__(self, config):
        self.path_task_dir = config['path']
        self.task = config['task']
        self.num_imgs = config['num_imgs']
        self.scale = config['params_pre']['scale']
        self.params_ext = config['params_ext']
        self.params_reg = config['params_reg']
        self.params_nontar = config['params_nontar']
        self.params_track = config['params_track']
        self.params_display = config['params_display']
        self.Hs, self.sus, self.tracks = [], [], []
        self.get_list_imgs(config['path'], config['postfix'])


    def get_list_imgs(self, path, postfix='.fits'):
        """Get all the image paths in the directory or the list of images"""
        self.postfix = postfix
        if isinstance(path, str):
            self.path_task_dir = path
            list_imgs = [os.path.join(self.path_task_dir, i) 
                              for i in os.listdir(self.path_task_dir) 
                              if i.endswith(postfix) and self.task in i]
        elif isinstance(path, list):
            self.path_task_dir = os.path.dirname(path[0])
            list_imgs = path
        if self.num_imgs == 'all':
            pass
        elif  isinstance(self.num_imgs, int):
            list_imgs = list_imgs[:self.num_imgs]
        elif isinstance(self.num_imgs, list):
            list_imgs = list_imgs[self.num_imgs[0]:self.num_imgs[1]]
        else:
            raise KeyError('num_imgs should be int or "all"!!!')
        self.list_imgs = sorted(list_imgs)  
        if list_imgs == []:
            raise KeyError('No images found in the directory!!!')


    def imread(self, path_img):
        """Read the image and return the image data and header"""
        if path_img.endswith('.fits') or path_img.endswith('.fit'):
            raw_picture=fits.open(path_img, ignore_missing_simple=True)   
            header = raw_picture[0].header             
            img = raw_picture[0].data
            return np.ascontiguousarray(img, np.float32), header
        elif path_img.endswith('.tif'):
            img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED), None
        else:   
            raise KeyError('wrong image format!!!')


    def preprocess(self, img):
        """Preprocess the image, including median filter and background subtraction"""
        img = cv2.medianBlur(img, 3)
        bkg = sep.Background(img)
        fore = img - np.array(bkg)
        h, w = img.shape
        nh, nw = int(h*self.scale), int(w*self.scale)
        self.h, self.w = nh, nw
        oup = cv2.resize(fore, (nw,nh))
        return oup
    

    def extract(self, img):
        """Extract the sources from the single image"""
        method = self.params_ext['method']
        if method == 'nth':
            coords = extract_NTH(img, self.params_ext['kernel_size'], self.params_ext['threshold'])
        elif method == 'sep':
            coords = extract_sep(img, self.params_ext['threshold'], self.params_ext['deblend'])
        else:
            raise KeyError('wrong extraction function!!!')
        h, w = img.shape
        border = self.params_ext['boundary']
        for coord in coords:
            if coord[0] < border or coord[0] > (w-border) or coord[1] < border or coord[1] > (h-border):
                coords = np.delete(coords, np.where(coords==coord), axis=0)
        return coords
    

    def register_img(self, img1, img2, flag_img=False):
        """Register the two images and return the homography matrix"""
        num_register_stars = self.params_reg['num_stars']
        ratio = self.params_reg['ratio_len']
        th_len = self.params_reg['th_length']
        ta1 = TriAngleRectifyWithDelaunay(img1, numStars=num_register_stars, ratio=ratio, thLen=th_len)
        ta2 = TriAngleRectifyWithDelaunay(img2, numStars=num_register_stars, ratio=ratio, thLen=th_len)
        H12, _ = ta1.getH(ta2)
        self.Hs.append(H12)
        if flag_img:
            img12 = cv2.warpPerspective(img1, H12, img1.shape[::-1])
            return H12, img12
        else: return H12
    

    def register_tracks(self, H12):
        """Register the tracks from the last axis to the current axis with the homography matrix"""
        for track in self.tracks:
            track.update(H12)
        for i, sus in enumerate(self.sus):
            self.sus[i] = registerPoints(sus, H12)

    
    def cal_angle(self, vec1, vec2):
        """Calculate the intersection angle between two tracks"""
        u = np.array(vec1)
        v = np.array(vec2)
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        cos_theta = dot_product / (norm_u * norm_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta = np.degrees(theta)
        return theta

    
    def filter_non_targets(self, pts1, pts2, H12):
        """Filter the non-targets in the current frame based on the last frame"""
        th_dis = self.params_nontar['distance_threshold']
        pts12 = registerPoints(pts1, H12)
        marks = np.zeros(pts2.shape[0])
        for i, pt in enumerate(pts2):
            dis = np.linalg.norm(pts12 - pt, ord=2, axis=1)
            if (dis < th_dis).sum(): marks[i] = 1
        pts_cur = pts2[marks==0]
        map_sus = np.zeros((self.h, self.w))
        for i, pt_cur in enumerate(pts_cur):
            x, y = int(pt_cur[0]+0.5), int(pt_cur[1]+0.5)
            map_sus[y, x] = i+1
        return pts_cur, map_sus    
    

    def get_intertime(self, header1, header2):
        """Get the interframe interval from the header"""
        if header1 is None or header2 is None:
            return 1
        t1 = header1['DATE-OBS']
        exp1 = header1['EXPTIME']
        obj1 = datetime.fromisoformat(t1)
        t2 = header2['DATE-OBS']
        exp2 = header2['EXPTIME']
        obj2 = datetime.fromisoformat(t2)
        diff_obj = obj2 - obj1
        delta_t = diff_obj.total_seconds() + (exp2-exp1)/2
        return delta_t
    

    def associate_tracks(self, idx, map_sus, pts_cur, delta_t):
        """Associate all the suspected points in the current frame to the history tracks"""
        r_search = self.params_track['association_radius']
        r_border = self.params_track['r_boundary']
        th_ang = self.params_track['angle_threshold']
        for i, track in enumerate(self.tracks):
            # filter false tracks
            pt_last = track.cur_pt
            if (pt_last.conf<0): continue
            if (track.cur_idx+1 != idx): continue
            # track predict
            xf, yf = pt_last.x+track.vx*delta_t, pt_last.y+track.vy*delta_t
            # stop to predict track over boundary
            if (xf<r_border) or (xf>self.w-r_border) or (yf<r_border) or (yf>self.h-r_border):
                continue
            map_roi = get_roi(map_sus, xf, yf, r_search, False, flatCopy=False)
            if map_roi.sum() > 0:  
                ys_roi, xs_roi = np.where(map_roi)
                ys, xs = ys_roi+yf-r_search, xs_roi+xf-r_search
                pos = np.argmin((xs-xf)**2+(ys-yf)**2)
                tar_idx = int(map_roi[ys_roi[pos], xs_roi[pos]])
                x_pred, y_pred = pts_cur[tar_idx-1]
                vx, vy = (x_pred-pt_last.x)/delta_t, (y_pred-pt_last.y)/delta_t
                conf = pt_last.conf+1
                ang = self.cal_angle((vx, vy), (track.vx, track.vy))
                if ang < th_ang:
                    map_roi[ys_roi[pos], xs_roi[pos]] = 0
                    pt_new = PT(idx, x_pred, y_pred, vx, vy, 1, conf)
                    track.append(pt_new)
                    continue
            pt_new = PT(idx, xf, yf, pt_last.vx, pt_last.vy, 0, pt_last.conf-1)
            track.append(pt_new)


    def initialize_tracks(self, idx, map_sus, pts_last, pts_cur, H12, delta_t):
        """Initialize the new tracks based on the suspected points in the current frame and last frame"""
        r_border = self.params_track['r_boundary']
        r_associate = self.params_track['search_radius']
        pts_base = pts_last.copy()
        pts_last = registerPoints(pts_last, H12)
        for i, pt_last in enumerate(pts_last):
            (x_last, y_last) = pt_last
            if (x_last<r_border) or (y_last<r_border) or \
                (x_last>self.w-r_border) or (y_last>self.h-r_border): continue
            roi_sus = get_roi(map_sus, x_last, y_last, r_associate, False, flatCopy=False)
            # search neighbor points
            if roi_sus.sum() == 0: continue
            ys, xs = np.where(roi_sus)
            for j in range(xs.size):
                tar_idx = int(roi_sus[ys[j], xs[j]])
                x, y = pts_cur[tar_idx-1]
                vx, vy = (x-x_last)/delta_t, (y-y_last)/delta_t
                x0, y0 = pts_base[i][0], pts_base[i][1]
                pt0 = PT(idx-1, x_last, y_last, vx, vy, 1, 1, x0, y0)
                pt1 = PT(idx, x, y, vx, vy, 1, 2)
                track = Track(pt0, pt1, H12)
                self.tracks.append(track)


    def find_true_tracks(self,):
        """Filter falae tracks with conditions"""
        th_len = self.params_track['length_threshold']
        th_conf = self.params_track['confidence_threshold']
        tracks_true = []
        for track_id in range(len(self.tracks)):
            track = self.tracks[track_id]
            flag_len = track.len > th_len
            flag_conf = track.conf > th_conf
            flag = flag_len & flag_conf
            if flag: 
                tracks_true.append(track)
        self.tracks_true = tracks_true
        print(f'{self.task}: Found {len(tracks_true)} true tracks!!!')


    def track_postprocess(self,):
        """Postprocess tracks including leak point interpolation and delete wrong points"""
        for track in self.tracks_true:
            track.remove_invalid_points()
            track.interpolate()


    def main(self):
        """Main function to process the image sequence"""
        # last frame & source extraction
        raw1, header1 = self.imread(self.list_imgs[0])
        img1 = self.preprocess(raw1)
        pts_last = self.extract(img1)
        for idx in tqdm(range(len(self.list_imgs)-1), desc=self.task):
            # current frame & source extraction
            raw2, header2 = self.imread(self.list_imgs[idx+1])
            img2 = self.preprocess(raw2)
            
            # image registration
            H12, img12 = self.register_img(img1, img2, flag_img=True)

            # tracks registration
            self.register_tracks(H12)

            # source extraction
            pts1 = self.extract(img1)
            pts2 = self.extract(img2)

            # filter non-targets
            pts_cur, map_sus = self.filter_non_targets(pts1, pts2, H12)

            if idx > 0: 
                # get interframe interval
                delta_t = self.get_intertime(header1, header2)

                # track association
                self.associate_tracks(idx, map_sus, pts_cur, delta_t)

                # track initilization
                self.initialize_tracks(idx, map_sus, pts_last, pts_cur, H12, delta_t)
                
            # upadte
            img1, header1 = img2, header2
            self.sus.append(pts_last)
            pts_last = pts_cur

        # track filter
        self.find_true_tracks()

        # track postprocess
        self.track_postprocess()

        # print & show
        self.show_and_save(img2)


    def show_and_save(self, img):
        if self.params_display['print']:
            self.print_tracks(self.tracks_true)
        fig, _ = show_img(img)
        self.show_tracks(self.tracks_true, fig)
        if self.params_display['display']:
            plt.show()
        plt.close('all')
        if self.params_display['export']:
            self.save_tracks(self.tracks_true)
            self.save_tracks_txt(self.tracks_true)

    def show_tracks(self, tracks: list[Track], fig=None):
        """Show and save the given tracks in the image"""
        # plt.ion()
        if fig is None:
            fig, axes = plt.subplots(1,1,figsize=(16,8))
        for track in tracks:
            track.show(fig)
        if len(tracks)<10: plt.legend()
        plt.xlim(0, self.w)
        plt.ylim(self.h, 0)
        if len(tracks): plt.tight_layout()
        path = f'results/{os.path.basename(self.path_task_dir)}_{self.task}_tracks.png'
        plt.savefig(path, dpi=500)
        print(f'All the tracks are drawn in {path}!!!')


    def print_tracks(self, tracks: list[Track]):
        """Print the information of the given tracks"""
        print(f"{self.task} tracks info: ")
        for i, track in enumerate(tracks):
            con = f'\ntrack-{i}:\n'
            for pt in track.traj:
                con += f'\tframe-{pt.idx:02d}  status:{pt.status}  conf:{pt.conf:02d}  pt:({pt.x:6.2f},{pt.y:6.2f})  '
                con += f'pt0:({pt.x0:6.2f},{pt.y0:6.2f})  v:({pt.vx:5.2f}, {pt.vy:5.2f})\n'
            print(con)
            
    
    def show_sus(self, ):
        """Show the suspected points in the image, only for debug"""
        plt.figure()
        for sus in self.sus:
            plt.plot(sus[:, 0], sus[:, 1], marker='o', ms=3, linestyle='none')
    

    def save_tracks(self, tracks:list[Track]):
        """Save the detection results in each frame and save images"""
        dir_save = os.path.join(self.path_task_dir, 'results', self.task)
        if not os.path.exists(dir_save): os.makedirs(dir_save)
        samples = {}
        for idx, path_img in enumerate(tqdm(self.list_imgs[1:], desc='Saving images')):
            img = self.preprocess(self.imread(path_img)[0])
            img_rgb = convert_16_to_RGB(img)
            for track_id, track in enumerate(tracks):
                for pt in track.traj:
                    if pt.idx == idx:
                        cv2.circle(img_rgb, (int(pt.x0), int(pt.y0)), 10, (0,0,255), 2)
                        cv2.putText(img_rgb, str(track_id+1), (int(pt.x0)-5, int(pt.y0)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                        if pt.status == 1:
                            roi = get_roi(img, pt.x0, pt.y0, 30, True, flatCopy=False)
                            if samples.get(track_id) is None:
                                samples[track_id] = []
                            else:
                                samples[track_id].append(roi)
            name_img = os.path.basename(path_img).replace(self.postfix, 'tif')
            path_save = os.path.join(dir_save, name_img)
            cv2.imwrite(path_save, img_rgb)
        # save samples to gif
        for track_id, imgs in samples.items():
            path_gif = os.path.join(dir_save, f'{self.task}_target{track_id+1}.gif')
            imageio.mimsave(path_gif, imgs, fps=8)
        print('Detection result in each frame are drawn and saved in ', dir_save, '\n\n')


    def save_tracks_txt(self, tracks:list[Track]):
        """Save the detection results in a csv file"""
        dir_save = os.path.join(self.path_task_dir, 'results', self.task)
        if not os.path.exists(dir_save): os.makedirs(dir_save)
        path_save = os.path.join(dir_save, f'{self.task}_tracks.csv')
        line = ''
        for idx in range(len(self.list_imgs)):
            line += f'{idx+1:04d}, {os.path.basename(self.list_imgs[idx])}, '
            result = {track_id: ' , ' for track_id in range(len(tracks))}
            for track_id, track in enumerate(tracks):
                for pt in track.traj:
                    if pt.idx + 1 == idx:
                        result[track_id] = f'{pt.x0:6.2f}, {pt.y0:6.2f}'
            for track_id, pt in result.items(): 
                line += f'{track_id+1:02d}, {pt}, '
            line += '\n'
        with open(path_save, 'w') as f:
            f.write('ID, IMG, TRACK_ID, x0, y0, TRACK_ID, x1, y1, ...\n')
            f.write(line)




