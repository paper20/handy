import numpy as np
import json

path_len = 24
sensor_len = 16

def path_pro(path_str):
    if type(path_str) is float:
        print('path float:',path_str)
        return [0] * path_len
    if not path_str.strip():
        print('path is empty:', path_str)
        return [0] * path_len
    path_return = []
    path_list = path_str.split('_')
    for path in path_list:
        if ',' in path:
            paths = path.split(',')
            paths = paths[0:2]
            path_return = path_return + paths
        else:
            path_return.append(path)
    if len(path_return) < path_len:
        path_return = path_return + [0]*(path_len-len(path_return))
    else:
        path_return = path_return[0:path_len]
    return path_return

def radius_pro(path_str):
    if type(path_str) is float:
        print('path float (radius):',path_str)
        return [0] * path_len
    if not path_str.strip():
        print('path is empty (radius):', path_str)
        return [0] * path_len
    path_return = []
    path_list = path_str.split('_')
    for path in path_list:
        if ',' in path:
            paths = path.split(',')
            paths = paths[2:4]
            path_return = path_return + paths
        else:
            path_return.append(path)
    if len(path_return) < path_len:
        path_return = path_return + [0]*(path_len-len(path_return))
    else:
        path_return = path_return[0:path_len]
    return path_return

def sensor_pro(sensor_str, sampling):
    '''
    from end to start
    '''
    if type(sensor_str) is float:
        print('sensor float:', sensor_str)
        return [0.0] * (sensor_len*3)
    if not sensor_str.strip():
        print('path is empty:', path_str)
        return [0.0] * (sensor_len*3)
    sx, sy, sz = [], [], []
    sensor_list = sensor_str.split('&')
    for sensor in sensor_list:
        if '_' in sensor:
            sensors = sensor.split('_')
            sx.append(float(sensors[0]))
            sy.append(float(sensors[1]))
            sz.append(float(sensors[2]))
        else:
            print('no _ in sensor:', sensor)
    if sampling:
        sx = sx[::10]
        sy = sy[::10]
        sz = sz[::10]
    if len(sx) < sensor_len:
        sx = [0.0]*(sensor_len - len(sx)) + sx
        sy = [0.0]*(sensor_len - len(sy)) + sy
        sz = [0.0]*(sensor_len - len(sz)) + sz 
    else:
        sx = sx[-sensor_len:]
        sy = sy[-sensor_len:]
        sz = sz[-sensor_len:]
    sensor_return = sx + sy + sz
    return sensor_return

def os_pro(device_model):
    device_model = device_model.lower()
    if ('iphone' in device_model) or ('ios' in device_model):
        return 0
    else:
        return 1

def calculate_angle(row, i):
    if i == 1:
        x_1, x_2, y_1, y_2 = 'x1', 'pathx_'+str(i), 'y1', 'pathy_'+str(i)
    elif 2 <= i <= int(path_len/2):
        x_1, x_2, y_1, y_2 = 'pathx_'+str(i-1), 'pathx_'+str(i), 'pathy_'+str(i-1), 'pathy_'+str(i)
    else:
        x_1, x_2, y_1, y_2 = 'pathx_'+str(i-1), 'x2', 'pathy_'+str(i-1), 'y2'

    if (row[x_1] != 0) and (row[x_2] != 0) and (row[y_1] != 0) and (row[y_2] != 0):
        return np.arctan2(row[x_1] - row[x_2], row[y_1] - row[y_2])
    elif (row[x_1] != 0) and (row[x_2] == 0) and (row[y_1]!= 0) and (row[y_2] == 0):
        return np.arctan2(row[x_1]-row['x2'], row[y_1]-row['y2'])
    else:
        return 0

def calculate_distance(row, i):
    if i == 1:
        x_1, x_2, y_1, y_2 = 'x1', 'pathx_'+str(i), 'y1', 'pathy_'+str(i)
    elif 2 <= i <= int(path_len/2):
        x_1, x_2, y_1, y_2 = 'pathx_'+str(i-1), 'pathx_'+str(i), 'pathy_'+str(i-1), 'pathy_'+str(i)
    else:
        x_1, x_2, y_1, y_2 = 'pathx_'+str(i-1), 'x2', 'pathy_'+str(i-1), 'y2'

    if (row[x_1] != 0) and (row[x_2] != 0) and (row[y_1] != 0) and (row[y_2] != 0):
        return np.sqrt((row[x_1]-row[x_2])**2 + (row[y_1]-row[y_2])**2)
    elif (row[x_1] != 0) and (row[x_2] == 0) and (row[y_1] != 0) and (row[y_2] == 0):
        return np.sqrt((row[x_1]-row['x2'])**2 + (row[y_1]-row['y2'])**2)
    else:
        return 0
    
def pre_process(df):

    # device_id from str to int
    df["device_id"] = df["device_id"].astype(int)

    # replace inf and null with 0
    for key in ["start_touch_position", "end_touch_position", "x1", "x2", "y1", "y2"]:
        df[key] = df[key].replace([np.inf, -np.inf], np.nan)
        df[key] = df[key].replace([np.nan], 0)
        df[key] = df[key].astype(int)

    # resolution str split
    df[["resolutionh", "resolutionw"]] = df["resolution"].str.split("*", expand=True)

    # replace inf and null with mean
    mean_dict = {}
    for key in ["duration", "resolutionh", "resolutionw", "screenh", "screenw", "speed"]:
        df[key] = df[key].astype(float)
        df[key] = df[key].replace([np.inf, -np.inf], np.nan)
        mean_value = np.mean(df[key])
        mean_value = round(mean_value, 4)
        mean_dict[key] = mean_value
        df[key] = df[key].replace([np.nan], mean_value)

    # label tag to label id, task: all-7
    label_map_7 = {
        'l_l': 0,
        'l_r': 1,
        'r_l': 2,
        'r_r': 3,
        'd_d': 4,
        'u_d': 5,
        'd_u': 6
    }
    df['label_7'] = df['label'].map(label_map_7)
    label_map_7_r = {value:key for key, value in label_map_7.items()}

    # label tag to label id, task: cradle-3
    label_map_cradle_3 = {
        'l_l': 0,
        'l_r': 0,
        'r_l': 1,
        'r_r': 1,
        'd_d': 2,
        'u_d': 2,
        'd_u': 2
    }
    df['label_3'] = df['label'].map(label_map_cradle_3)

    # label tag to label id, task: slide-2
    label_map_slide_2 = {
        'l_l': 0,
        'l_r': 1,
        'r_l': 0,
        'r_r': 1
    }
    unknown_value = 2
    df['label_2'] = df['label'].map(label_map_slide_2).fillna(unknown_value)

    # slide direction
    driction_map = {
        "down": 0,
        "up": 1,
        "left": 2,
        "right": 3
    }
    df["slide_direction"] = df["slide_direction"].map(driction_map)

    # device model
    df['device_model'] = df['device_model'].apply(lambda x : os_pro(x))

    # path x and path y
    pathx = ['pathx_'+str(i) for i in range(1,int(path_len/2)+1)]
    pathy = ['pathy_'+str(i) for i in range(1,int(path_len/2)+1)]
    paths = [item for pair in zip(pathx, pathy) for item in pair]
    df[paths] = df['path'].apply(lambda x: path_pro(x)).tolist()
    for path in paths:
        df[path] = df[path].fillna(0)
        df[path] = df[path].astype(int)

    # radius x and radius y
    radiusx = ['radiusx_'+str(i) for i in range(1,int(path_len/2)+1)]
    radiusy = ['radiusy_'+str(i) for i in range(1,int(path_len/2)+1)]
    radii = [item for pair in zip(radiusx, radiusy) for item in pair]
    df[radii] = df['path'].apply(lambda x: radius_pro(x)).tolist()
    for radius in radii:
        df[radius] = df[radius].fillna(0.0)
        df[radius] = df[radius].astype(float)
    
    # distance and angle
    df["y2_y1"] = df["y2"] - df["y1"]
    df["x2_x1"] = df["x2"] - df["x1"]
    df['angle'] = np.arctan2(df['x1'] - df['x2'], df['y1'] - df['y2'])
    df['distance'] = np.sqrt((df['x2'] - df['x1'])**2 + (df['y2'] - df['y1'])**2)
    
    for i in range(1, int(path_len/2)+2):
        df['angle_'+str(i)] =  df.apply(calculate_angle, axis=1, i=i)
    
    for i in range(1, int(path_len/2)+2):
        df['distance_'+str(i)] =  df.apply(calculate_distance, axis=1, i=i)
    
    for i in range(1, int(path_len/2)+2):
        if i == 1:
            df['real_distance']  = df['distance_'+str(i)]
        else:
            df['real_distance'] = df['real_distance'] + df['distance_'+str(i)]
    
    # motion sensor transformation
    sensor_keys = [ "gravity", "gyroscope", "linearacceleration", "ryp", "real_gravity", "real_gyroscope", "real_linearacceleration", "real_ryp", "live_gravity", "live_gyroscope", "live_linearacceleration", "live_ryp"]
    zscore_dict = {}
    sampling = False
    for sensor in sensor_keys:
        sensorx = [sensor+"_x_"+str(i) for i in range(1,sensor_len+1)]
        sensory = [sensor+"_y_"+str(i) for i in range(1,sensor_len+1)]
        sensorz = [sensor+"_z_"+str(i) for i in range(1,sensor_len+1)]
        # sensor3 = [item for pair in zip(sensorx, sensory, sensorz) for item in pair]
        sensor3 = sensorx + sensory + sensorz
        if 'real' in sensor:
            sampling = True
        df[sensor3] = df[sensor].apply(lambda x: sensor_pro(x, sampling)).tolist()
        x_mean, x_std = df[sensorx].values.flatten().mean(), df[sensorx].values.flatten().std()
        x_mean, x_std = round(x_mean,4), round(x_std,4)
        for key in sensorx:
            df[key] = df[key].fillna(0)
            df[key] = df[key].astype(float)
            df[key] = (df[key] - x_mean)/x_std
        y_mean, y_std = df[sensory].values.flatten().mean(), df[sensory].values.flatten().std()
        y_mean, y_std = round(y_mean,4), round(y_std,4)
        for key in sensory:
            df[key] = df[key].fillna(0)
            df[key] = df[key].astype(float)
            df[key] = (df[key] - y_mean)/y_std
        z_mean, z_std = df[sensorz].values.flatten().mean(), df[sensorz].values.flatten().std()
        z_mean, z_std = round(z_mean,4), round(z_std,4)
        for key in sensorz:
            df[key] = df[key].fillna(0)
            df[key] = df[key].astype(float)
            df[key] = (df[key] - z_mean)/z_std
        zscore_dict[sensor] = {
            "x":{
                "mean": x_mean,
                "std": x_std
                },
            "y":{
                "mean": y_mean,
                "std": y_std
                },
            "z":{
                "mean": z_mean,
                "std": z_std
                }
        }

    return df
