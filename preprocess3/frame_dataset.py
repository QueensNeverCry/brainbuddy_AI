import os
from tkinter import Image


class VideoFrameDataset(Datatset):
    def __init__(self, data_list, transform = None, verbose=True):
        data_list=[]
        self.transform = transform
        self.verbose= verbose

        for folder_path, label in data_list :
            if not os.path.isdir(folder_path): # 폴더가 해당 경로에 없는 경우 : 건너뛰기
                continue
            self.data_list.append((folder_path,label))
        
        if self.verbose :
            print(f"유효한 샘플 수 : {len(self.data_list)}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        folder_path, label = self.data_list[idx]

        img_files = sorted([
            fname for fname in os.listdir(folder_path)
            if fname.lower().endswith(('.jpg','.jpeg','.png'))
        ])
        selected_files = img_files[:30]
        frames =[]

        for fname in selected_files:
            img_path = os.path.join(folder_path,fname)
            image = Image.open(img_path).convert("RGB")
            if self.transform :
                image = self.transform(image)
            
            video = torch.stack(frames)