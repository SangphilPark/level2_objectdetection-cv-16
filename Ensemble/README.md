## 사용법

```
#csv 파일 경로 설정
csv_path1 = '경로1'
csv_path2 = '경로2'
csv_path3 = '경로3'

csv_data1 = pd.read_csv(csv_path1,keep_default_na=False)
csv_data2 = pd.read_csv(csv_path2,keep_default_na=False)
csv_data3 = pd.read_csv(csv_path3,keep_default_na=False)

csv_data = [csv_data1, csv_data2,csv_data3]

save_dir = './'

weights = [1, 1, 1]

iou_thr = 0.5
skip_box_thr = 0.0001
sigma = 0.1
```

- pip install ensemble-boxes 로 library 설치

- save_dir
  - 저장할 경로
- ensemble 방식
  - **ensemble 방식은 parameter로 넘겨주진 않고 코드에 함수 내부에 따로 주석처리 해놨습니다!** default는 weighted box fusion으로 되어있고 예를 들어 nms를 사용하고 싶으시면 주석처리 풀고 wbf는 주석처리 하시면 돌아갑니다!
- csv_data list
  - 앙상블하려는 model의 output csv 파일들을 불러와 주시고 **1개의 list**로 만들어주시면 됩니다.
- weights
  - 앙상블 하려는 **모델의 갯수만큼 weight를 지정**해주시면 됩니다. 예를 들어 3개 모델을 앙상블하고 동일하게 가중치를 주고 싶으시면 [1,1,1]로 넘겨주시면 됩니다.

## 참고 repo

[WBF(weighted box fusion)](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
