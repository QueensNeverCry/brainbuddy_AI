# gen_requirements_from_freeze.py
# freeze.txt에서 필요한 패키지만 뽑아 exact 버전으로 requirements.txt 생성
# 대상: torch, torchvision, numpy, pillow, opencv-python, tqdm, mediapipe (옵션)
# (학습/라벨링까지 백엔드가 돌릴 거면 pandas, scikit-learn, open_clip_torch도 포함)

import re

# 필요한 패키지 목록 (소문자)
CORE = {
    "torch",
    "torchvision",
    "numpy",
    "pillow",
    "opencv-python",
    "tqdm",
    "mediapipe",          # 설치 어려우면 빼도 됨 (Haar fallback)
}
# 라벨링/학습까지 서버에서 쓸 경우 주석 해제
# CORE.update({"pandas", "scikit-learn", "open_clip_torch", "matplotlib"})

# freeze.txt 읽기
with open("freeze.txt", "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip() and "@" not in ln]  # VCS/로컬 링크는 제외

# pkg==ver 라인만 남기고, 패키지명 소문자로 정규화한 dict 구성
pin = {}
for ln in lines:
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s]+)$", ln)
    if not m: 
        continue
    name, ver = m.group(1), m.group(2)
    lname = name.lower()
    # mediapipe-silicon 등 하위 변형 이름 처리
    if lname.startswith("mediapipe"):
        lname = "mediapipe"
        name  = "mediapipe"
    pin[lname] = f"{name}=={ver}"

# 필요한 패키지만 추출
missing = []
out = []
for pkg in sorted(CORE):
    if pkg in pin:
        out.append(pin[pkg])
    else:
        missing.append(pkg + " (not found in freeze.txt)")

# 파일 쓰기
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write("# pinned from your local environment\n")
    for ln in out:
        f.write(ln + "\n")

print("✔ wrote requirements.txt")
if missing:
    print("⚠ not found in freeze.txt:", ", ".join(missing))
