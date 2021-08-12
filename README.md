# ColorAid
![image](https://user-images.githubusercontent.com/52124204/129129543-b0aff252-3772-47f8-8d22-0d2c313c20d2.png)

## Description

선화를 넣으면 밑색을 자동 채색해주는 웹 어플리케이션 입니다.

현재 미완성이며 밑색 자동 채색 기능 하기 전 영역 구분을 해주는 segmentation까지만 수행합니다.



## Environment

1. 먼저 프로그램을 실행하기 위해서 Python 3.8 환경이 필요합니다. 그리고 repository를 clone합니다.

```
https://git.swmgit.org/swm-12/12_swm40/Null-AI.git
```

2. 의존성 라이브러리를 설치합니다.

```
pip install awscli==1.20.14
pip install boto3==1.18.14
pip install Flask==2.0.1
pip install matplotlib==3.4.2
pip install numpy==1.21.1
pip install opencv-python==4.5.3.56
pip install scikit-image==0.18.2
```

3. `../web_app/config/s3_config.py`을 작성합니다

```
AWS_ACCESS_KEY = "S3 엑세스 키"
AWS_SECRET_ACCESS_KEY = "S3 시크릿 엑세스 키"
AWS_S3_BUCKET_REGION = "S3 리전 이름"
AWS_S3_BUCKET_NAME = "S3 버킷 이름"
```

4. 마지막으로 `../web_app/templates/index.html`의 S3 버킷 URL을 수정합니다.

```
...
<img src="S3 버킷 URL" alt="" width="{{ w }}" height="300">
...
```



## Usage

1. Flask app을 실행하기 위해서 `main.py`를 실행합니다.

```
python main.py
```

2.  브라우저에서 `http://localhost:8080`에 접속합니다.

![image](https://user-images.githubusercontent.com/52124204/129131212-296000a5-5adb-4c41-bffd-7851c9f2e4f1.png)

3. `파일 선택`을 눌러 채색할 이미지를 선택하고 `업로드하고 채색`을 누릅니다.

![image](https://user-images.githubusercontent.com/52124204/129131295-a32552d7-a4b9-4ae1-aec3-bdc33f874d56.png)

4. 채색된 결과를 확인하고 사용합니다.



## Experiment

아래는 Segmentation 관련하여 실험한 내용입니다.

### 1. cv2.watershed()

![image](https://user-images.githubusercontent.com/52124204/129132002-891f5f25-12cd-4cb2-9154-49f00553e9ee.png)

### 2. skimage.segmentation.watershed()

![image](https://user-images.githubusercontent.com/52124204/129132009-5cdbdce6-a75f-415e-b63a-fb866a5ab1ef.png)

### 3. Felzenszwalb, P.F., Huttenlocher, D.P.: Efficient graph-based image segmentation. IJCV (2004)

![image](https://user-images.githubusercontent.com/52124204/129132019-c8dee5d3-be80-4c58-a824-1141e77abe82.png)

### 4. Lvmin Zhang, Yi JI, and Chunping Liu: DanbooRegion: An Illustration Region Dataset. ECCV (2020)

논문에서 제공하는 Pre-trained 모델을 실행하여 결과만 확인하였습니다.

![image](https://user-images.githubusercontent.com/52124204/129132133-2db0e122-f3e2-42fa-b699-164a0d253930.png)

