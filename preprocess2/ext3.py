import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_base_path, segment_duration=10, target_fps=30, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"영상 열기 실패 :{video_path}")
        return
    
    # 영상 fps가 불확실한 경우 사용. 현재 데이터셋은 30fps
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # if fps <= 0:
    #     print(f"⚠️ FPS 정보를 가져올 수 없습니다: {video_path}")
    #     cap.release()
    #     return
    
    fps =30
    frame_interval = max(int(fps/target_fps),1)
    segment_frame_count = segment_duration * fps # 300frames (한 segment 안의 frame 개수)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#영상이 총 몇프레임인지 가져오기
    num_segments = total_frames // segment_frame_count
    
    print(f"🎬 세그먼트 수: {num_segments}, Interval: {frame_interval}프레임마다 저장")
    os.makedirs(output_base_path,exist_ok=True)
    saved_total=0
    
    for segment_idx in tqdm(range(num_segments),desc="300프레임 단위로 분리"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx*segment_frame_count)# 해당 프레임 위치로 이동(점프)
       
        count=0
        saved=0
        retry_count = 0
        while saved<max_frames:
            current_frame_pos=cap.get(cv2.CAP_PROP_POS_FRAMES)#현재 비디오에서 읽고 있는 프레임 번호 가져오기
            if current_frame_pos >=(segment_idx+1)* segment_frame_count :
                break
            
            ret, frame = cap.read()
            if not ret:
                if retry_count < 2:
                    retry_count += 1
                    print(f"⚠️ 세그먼트 {segment_idx} 프레임 읽기 실패, 재시도 {retry_count}/{2}...")
                    # 프레임 위치 재조정 후 재시도
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    continue
                else:
                    print(f"❌ 세그먼트 {segment_idx} 프레임 읽기 {2}회 실패, 해당 세그먼트 건너뜀.")
                    break  # 재시도 초과하면 세그먼트 중단
            
            retry_count = 0  # 성공했으니 재시도 카운트 초기화
            
            if count % frame_interval ==0 :
                frame_path = os.path.join(output_base_path, f"{saved:04d}.jpg")
                cv2.imwrite(frame_path,frame)
                saved +=1
                
            count+=1
        print(f"📁 세그먼트 {segment_idx} 완료 ({saved}장 저장)")
    cap.release()
    print("전체 작업 완료")
        
if __name__ == "__main__":
    video_folder=f"D:/AIhub/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_20_02_1"
    output_root =f"D:/AIhub_frames"
    
    video_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(".mp4")
    ])
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # 파일명에서 확장자 제거
        output_base_path = os.path.join(output_root, video_name)

        extract_frames(video_path, output_base_path)
        print(f"================= {video_file} 전처리 완료 ===================\n")