import csv
import os
from tqdm import tqdm
import pickle

# trainLabels에 없는 영상들이 trainData에 있는 문제 발생
# all labels 에서 해당 영상 클립ID 에 해당하는 라벨을 가져옴
def load_labels(label_csv_path1,label_csv_path2):
    """CSV 파일에서 ClipID -> label 매핑 딕셔너리 생성"""
    label_dict1={}
    with open(label_csv_path1, newline="") as csvfile1:
        reader =csv.DictReader(csvfile1)
        for row in reader:
            clip_id = row["ClipID"]
            label = int(row["binary_label"]) 
            label_dict1[clip_id]=label
    label_dict2={}
    
    with open(label_csv_path2, newline="") as csvfile2:
        reader =csv.DictReader(csvfile2)
        for row in reader:
            clip_id = row["ClipID"]
            label = int(row["binary_label"])
            label_dict2[clip_id]=label
    return label_dict1, label_dict2

def match_train_and_label(train_txt_path, data_root,label_csv_path1,label_csv_path2):
    label_dict1,label_dict2 = load_labels(label_csv_path1,label_csv_path2)
    dataset_link=[]
    
    with open(train_txt_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    for filename in tqdm(filenames,desc="영상처리 중..."):
        file_id = os.path.splitext(filename)[0]  # "1100062016"
        user_id = file_id[:6]  # 앞 6자리: 사용자 ID "110006"

        # 저장 경로 -> 절대 경로로 수정
        output_dir = os.path.join(data_root, user_id, file_id)
        output_dir = os.path.normpath(output_dir)
        
        #여기서 이제 리스트[(튜플)]형태로 : (output_dir, label 로 저장)
        label = label_dict1.get(file_id)
        
        if label is None:#라벨값이 없으면 AllLabels에서 찾기
            label = label_dict2.get(file_id)
        if label is not None: # 있으면 append
            dataset_link.append((output_dir, label))
        else:
            print(f"⚠️ 라벨 없음: {file_id}")

    print(dataset_link[:5])      
    return dataset_link

if __name__ == "__main__":
    train_txt_path = "../DataSet/Train.txt"
    valid_txt_path = "../DataSet/Validation.txt"

    save_train = "C:/KSEB/brainbuddy_AI/frames/train_frames"
    save_valid = "C:/KSEB/brainbuddy_AI/frames/valid_frames"
    
    train_label_csv = "./pre_labels/pre_TrainLabels.csv"
    val_label_csv = "./pre_labels/pre_ValidationLabels.csv"
    label_csv_path2 = "./pre_labels/pre_AllLabels.csv"

    traindataset_link = match_train_and_label(train_txt_path, save_train, train_label_csv, label_csv_path2)
    valdataset_link = match_train_and_label(valid_txt_path, save_valid, val_label_csv, label_csv_path2)
    # 예: [('extracted_frames/110006/1100062016', 1), ('extracted_frames/110007/1100073012', 0), ...]
    
    with open("train_link.pkl", "wb") as f:
        pickle.dump(traindataset_link, f)
    with open("val_link.pkl", "wb") as f:
        pickle.dump(valdataset_link, f)