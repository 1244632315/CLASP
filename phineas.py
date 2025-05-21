import os
import cv2
import sep
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from tool import load_img, show_img, draw3D, ax_imshow, get_fileList, get_roi, convert_16_to_RGB
from extract import extract, extract_sep, extract_NTH, nms
from register import TriAngleRectifyWithDelaunay, registerPoints


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
    def __init__(self, pt1, pt2):
        self.len = 2
        self.idx0 = pt1.idx
        self.cur_idx = pt2.idx
        self.register_idx = pt2.idx
        self.cur_pt = pt2
        self.traj = [pt1, pt2]
        self.update_speed()
        self.conf = pt2.conf

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

    def update(self, H):
        for i, pt in enumerate(self.traj):
            (xr, yr) = registerPoints((pt.x, pt.y), H)
            pt.x, pt.y = xr, yr
    
    def update_speed(self, ):
        vxs = self.get('vx')
        vys = self.get('vy')
        self.vx = np.mean(vxs)
        self.vy = np.mean(vys)
    
    def update_conf(self, ):
        confs = self.get('status')
        self.conf = sum(confs)

    def show(self, fig=None):
        if fig is None:
            plt.figure()
        xs, ys = self.get('x'), self.get('y')
        lab = f'{self.idx0} - {self.cur_idx}'
        plt.plot(xs, ys, ms=2, label=lab, marker='o', linestyle='-')


class DBT(object):
    def __init__(self, path, task, postfix='fits', scale=0.2, method='SEP'):
        # params filtering stars
        self.th_dis = 5

        # params for association
        self.r_associate = 30
        self.r_search = 10
        self.r_border = 20
        self.th_ang = 25

        # init
        self.task = task
        self.get_list_imgs(path, postfix)
        self.scale = scale
        self.method = method
        self.init_params()


    def init_params(self, ):
        self.Hs = []
        self.tracks = []
        self.sus = []


    def get_list_imgs(self, path, postfix='.fits'):
        self.postfix = postfix
        if isinstance(path, str):
            self.path_task_dir = path
            self.list_imgs = [os.path.join(self.path_task_dir, i) for i in os.listdir(self.path_task_dir) if i.endswith(postfix)]
        elif isinstance(path, list):
            self.path_task_dir = os.path.dirname(path[0])
            self.list_imgs = path

    def preprocess(self, img):
        img = cv2.medianBlur(img, 3)
        bkg = sep.Background(img)
        fore = img - np.array(bkg)
        h, w = img.shape
        nh, nw = int(h*self.scale), int(w*self.scale)
        self.h, self.w = nh, nw
        oup = cv2.resize(fore, (nw,nh))
        return oup
    

    def extract(self, img):
        if self.method == 'NTH':
            func = extract_NTH
        elif self.method == 'SEP':
            func = extract_sep
        else:
            raise KeyError('wrong extraction function!!!')
        coords = func(img)
        return coords
    

    def register_img(self, img1, img2, num_register_stars=25, ratio=0.05, flag_img=False):
        ta1 = TriAngleRectifyWithDelaunay(img1, numStars=num_register_stars, ratio=ratio)
        ta2 = TriAngleRectifyWithDelaunay(img2, numStars=num_register_stars, ratio=ratio)
        H12, _ = ta1.getH(ta2)
        self.Hs.append(H12)
        if flag_img:
            img12 = cv2.warpPerspective(img1, H12, img1.shape[::-1])
            return H12, img12
        else:
            return H12
    

    def register_tracks(self, H):
        for track in self.tracks:
            track.update(H)
        for i, sus in enumerate(self.sus):
            self.sus[i] = registerPoints(sus, H)

    
    def cal_angle(self, vec1, vec2):
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
        pts12 = registerPoints(pts1, H12)
        marks = np.zeros(pts2.shape[0])
        for i, pt in enumerate(pts2):
            dis = np.linalg.norm(pts12 - pt, ord=2, axis=1)
            if (dis < self.th_dis).sum(): marks[i] = 1
        pts_cur = pts2[marks==0]
        map_sus = np.zeros((self.h, self.w))
        for i, pt_cur in enumerate(pts_cur):
            x, y = int(pt_cur[0]+0.5), int(pt_cur[1]+0.5)
            map_sus[y, x] = i+1
        return pts_cur, map_sus    
    

    def associate_tracks(self, idx, map_sus, pts_cur):
        for i, track in enumerate(self.tracks):
            # filter false tracks
            pt_last = track.cur_pt
            if (pt_last.conf<0): continue
            # track predict
            xf, yf = pt_last.x+track.vx, pt_last.y+track.vy
            map_roi = get_roi(map_sus, xf, yf, self.r_search, False, flatCopy=False)
            if map_roi.sum() > 0:  
                ys_roi, xs_roi = np.where(map_roi)
                ys, xs = ys_roi+yf-self.r_search, xs_roi+xf-self.r_search
                pos = np.argmin((xs-xf)**2+(ys-yf)**2)
                tar_idx = int(map_roi[ys_roi[pos], xs_roi[pos]])
                x_pred, y_pred = pts_cur[tar_idx-1]
                vx, vy = x_pred-pt_last.x, y_pred-pt_last.y
                conf = pt_last.conf+1
                ang = abs(self.cal_angle((vx, vy), (track.vx, track.vy)))
                if ang < self.th_ang:
                    map_roi[ys_roi[pos], xs_roi[pos]] = 0
                    pt_new = PT(idx, x_pred, y_pred, vx, vy, 1, conf)
                    track.append(pt_new)
            else:
                pt_new = PT(idx, xf, yf, pt_last.vx, pt_last.vy, 0, pt_last.conf-1)
                track.append(pt_new)


    def initialize_tracks(self, idx, map_sus, pts_last, pts_cur, H12):
        pts_base = pts_last.copy()
        pts_last = registerPoints(pts_last, H12)
        for i, pt_last in enumerate(pts_last):
            (x_last, y_last) = pt_last
            if (x_last<self.r_border) or (y_last<self.r_border) or \
                (x_last>self.w-self.r_border) or (y_last>self.h-self.r_border): continue
            roi_sus = get_roi(map_sus, x_last, y_last, self.r_associate, False, flatCopy=False)
            # search neighbor points
            if roi_sus.sum() == 0: continue
            ys, xs = np.where(roi_sus)
            for j in range(xs.size):
                tar_idx = int(roi_sus[ys[j], xs[j]])
                x, y = pts_cur[tar_idx-1]
                vx, vy = x-x_last, y-y_last
                x0, y0 = pts_base[i][0], pts_base[i][1]
                pt0 = PT(idx-1, x_last, y_last, vx, vy, 1, 1, x0, y0)
                pt1 = PT(idx, x, y, vx, vy, 1, 2)
                track = Track(pt0, pt1)
                self.tracks.append(track)


    def main(self):
        # last frame & source extraction
        img1 = self.preprocess(load_img(self.list_imgs[0]))
        pts_last = self.extract(img1)
        for idx in tqdm(range(len(self.list_imgs)-1), desc=self.task):
            # current frame & source extraction
            img2 = self.preprocess(load_img(self.list_imgs[idx+1]))
            
            # image registration
            H12, img12 = self.register_img(img1, img2, flag_img=True, ratio=0.08)

            # tracks registration
            self.register_tracks(H12)

            # source extraction
            pts1 = self.extract(img1)
            pts2 = self.extract(img2)

            # filter non-targets
            pts_cur, map_sus = self.filter_non_targets(pts1, pts2, H12)

            if idx > 0: 
                # track association
                self.associate_tracks(idx, map_sus, pts_cur)

                # track initilization
                self.initialize_tracks(idx, map_sus, pts_last, pts_cur, H12)
                
            # upadte
            img1 = img2
            self.sus.append(pts_last)
            pts_last = pts_cur

        # track filter
        self.find_true_tracks()

        # print & show
        # self.print_tracks(self.tracks_true)
        plt.ion()
        fig, _ = show_img(img2)
        self.show_tracks(self.tracks_true, fig)
        # plt.show()
        plt.close('all')

        # save
        # self.save_tracks(self.tracks_true)
        self.save_tracks_txt(self.tracks_true)


    def find_true_tracks(self, th_len=3, th_conf=5):
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


    def show_tracks(self, tracks: list[Track], fig=None):
        # plt.ion()
        if fig is None:
            fig, axes = plt.subplots(1,1,figsize=(16,8))
        for track in tracks:
            track.show(fig)
        if len(tracks)<10: plt.legend()
        plt.xlim(0, self.w)
        plt.ylim(self.h, 0)
        plt.tight_layout()
        plt.savefig(f'results/{self.task}_tracks.png', dpi=500)


    def print_tracks(self, tracks: list[Track]):
        print(f"{self.task} tracks info: ")
        for i, track in enumerate(tracks):
            con = f'\ntrack-{i}:\n'
            for pt in track.traj:
                con += f'\tframe-{pt.idx:02d}  status:{pt.status}  conf:{pt.conf:02d}  pt:({pt.x:6.2f},{pt.y:6.2f})  '
                con += f'pt0:({pt.x0:6.2f},{pt.y0:6.2f})  v:({pt.vx:5.2f}, {pt.vy:5.2f})\n'
            print(con)
            
    
    def show_sus(self, ):
        plt.figure()
        for sus in self.sus:
            plt.plot(sus[:, 0], sus[:, 1], marker='o', ms=3, linestyle='none')
    

    def save_tracks(self, tracks:list[Track]):
        dir_save = os.path.join(self.path_task_dir, 'results', self.task)
        if not os.path.exists(dir_save): os.makedirs(dir_save)
        for idx, path_img in enumerate(self.list_imgs[1:]):
            img = self.preprocess(load_img(path_img))
            img_rgb = convert_16_to_RGB(img)
            for track in tracks:
                for pt in track.traj:
                    if pt.idx == idx:
                        cv2.circle(img_rgb, (int(pt.x0), int(pt.y0)), 10, (0,0,255), 2)
            name_img = os.path.basename(path_img).replace(self.postfix, 'tif')
            path_save = os.path.join(dir_save, name_img)
            cv2.imwrite(path_save, img_rgb)
        print('Results sucessfully saved in ', dir_save)
    

    def save_tracks_txt(self, tracks:list[Track]):
        dir_save = os.path.join(self.path_task_dir, 'results', self.task)
        if not os.path.exists(dir_save): os.makedirs(dir_save)
        path_save = os.path.join(dir_save, f'{self.task}_tracks.csv')
        line = ''
        for idx in range(len(self.list_imgs)):
            line += f'{idx+1:04d}, {os.path.basename(self.list_imgs[idx])}, '
            for track_id, track in enumerate(tracks):
                for pt in track.traj:
                    if pt.idx + 1 == idx:
                        line += f'{track_id+1:02d}, {pt.x0:6.2f}, {pt.y0:6.2f}'
            line += '\n'
        with open(path_save, 'w') as f:
            f.write('ID, IMG, TRACK_ID, x0, y0, TRACK_ID, x1, y1, ...\n')
            f.write(line)
        print('Results sucessfully saved in ', path_save)
        


def process_CLASP_seqs(dir_path):
    # classify the task data
    paths = get_fileList(dir_path)
    tasks = {}
    for path in paths:
        name = os.path.basename(path)
        keys = name.split('-')
        sky, cam = keys[:2]
        if sky not in tasks.keys():
            tasks[sky] = []
        tasks[sky].append(path)
    
    # process each task independently
    for task, paths in tasks.items():
        print(f'\nStart to process task: {task}')
        detector = DBT(paths, task=task, postfix='fits', scale=0.2, method='SEP')
        detector.main()



if __name__ == '__main__':
    detector = DBT('./imgs/Phineas_2', task='test1', scale=0.5)
    detector.main()