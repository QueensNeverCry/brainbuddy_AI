import json
import os
import numpy as np
import pandas as pd

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_features(anno):
    features = {}
    
    # Head Pose
    head = anno['pose']['head']
    features['head_yaw'], features['head_pitch'], features['head_roll'] = head

    # Camera distance
    features['cam_distance'] = anno['distance']['cam']

    # Gaze vector: point - cam
    cam = np.array(anno['pose']['cam'])
    point = np.array(anno['pose']['point'] + [0])
    gaze_vec = point - cam
    gaze_vec = gaze_vec / (np.linalg.norm(gaze_vec) + 1e-6)
    features['gaze_x'], features['gaze_y'], features['gaze_z'] = gaze_vec.tolist()

    # Eye centers
    for ann in anno['annotations']:
        if ann['label'] == 'l_center':
            features['l_eye_x'], features['l_eye_y'] = ann['points'][0]
        elif ann['label'] == 'r_center':
            features['r_eye_x'], features['r_eye_y'] = ann['points'][0]

    # EAR from eyelids
    def calc_EAR(pts):
        pts = np.array(pts)
        if len(pts) < 12:
            return 0
        v_dist = np.linalg.norm(pts[3] - pts[11])
        h_dist = np.linalg.norm(pts[0] - pts[8])
        return v_dist / (h_dist + 1e-6)

    for ann in anno['annotations']:
        if ann['label'] == 'l_eyelid':
            features['l_EAR'] = calc_EAR(ann['points'])
        elif ann['label'] == 'r_eyelid':
            features['r_EAR'] = calc_EAR(ann['points'])

    return features

def compute_dynamic_features(df):
    gaze = df[["gaze_x", "gaze_y", "gaze_z"]].values
    head_pose = df[["head_yaw", "head_pitch", "head_roll"]]

    # Gaze stability (cosine similarity)
    cos_sim = [1.0]
    for i in range(1, len(df)):
        sim = np.dot(gaze[i], gaze[i-1]) / (np.linalg.norm(gaze[i]) * np.linalg.norm(gaze[i-1]) + 1e-6)
        cos_sim.append(sim)
    df["gaze_stability"] = cos_sim

    # Head stability (rolling stddev)
    df["head_stability"] = head_pose.rolling(window=5, min_periods=1).std().mean(axis=1)

    # Gaze jitter
    gaze_std = pd.DataFrame(gaze).rolling(window=5, min_periods=1).std()
    df["gaze_jitter"] = gaze_std.mean(axis=1)

    # Gaze direction (z > 0.5)
    df["gaze_direction"] = (df["gaze_z"] > 0.5).astype(int)

    # Fixation duration
    fixation = []
    count = 0
    for s in df["gaze_stability"]:
        if s > 0.95:
            count += 1
        else:
            count = 0
        fixation.append(count)
    df["fixation_duration"] = fixation

    # Saccade frequency
    gaze_diff = np.linalg.norm(np.diff(gaze, axis=0), axis=1)
    gaze_diff = np.insert(gaze_diff, 0, 0)
    saccade = pd.Series(gaze_diff > 0.1).rolling(window=5, min_periods=1).mean()
    df["saccade_frequency"] = saccade

    # Blink detection
    df["blink"] = ((df["l_EAR"] + df["r_EAR"]) / 2 < 0.21).astype(int)

    # Distance change
    df["distance_change"] = df["cam_distance"].diff().fillna(0)

    # Head tilt
    df["head_tilt_angle"] = np.degrees(np.arctan2(df["head_pitch"], df["head_roll"] + 1e-6))

    return df

def process_json_folder(folder_path, save_csv=True, save_npy=False, output_prefix="output"):
    all_features = []

    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
    for file in json_files:
        data = load_json(os.path.join(folder_path, file))
        anno = data["Annotations"]
        features = extract_features(anno)
        all_features.append(features)

    df = pd.DataFrame(all_features)
    df = compute_dynamic_features(df)

    if save_csv:
        df.to_csv(f"{output_prefix}_features.csv", index=False)
    if save_npy:
        np.save(f"{output_prefix}_features.npy", df.to_numpy())

    print(f"Processed {len(df)} frames. Saved to '{output_prefix}_features.csv'")

