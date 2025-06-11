###########################
### obtain target detection results from local files ###
###########################



import pandas as pd
path_csv = './imgs/flux/results/SKYMAPPER0030-CAM1/SKYMAPPER0030-CAM1_tracks.csv'
df = pd.read_csv(path_csv)


class Track:
    def __init__(self, path_csv):
        self.data = pd.read_csv(path_csv)

    def query_frame(self, frame):
        if frame.isinstance(str):
            mask = self.data['IMG'] == frame
            raw = self.data[mask]
            track = []
            for track_id in range(1, 10):
                if raw['T'+track_id].isna: 
                    continue
                else:
                    x, y = float(raw['x'+track_id].values[0]), float(raw['y'+track_id].values[0])
                    track.append([track_id, x , y])
                    