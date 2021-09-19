# ColorAid AI Server
소프트웨어 마에스트로 12기 NULL팀의 웹툰 자동채색 어플리케이션 _ColorAid_의 AI Server입니다.

선화 이미지의 S3 URL을 요청으로 받으면 그 이미지를 받아서 밑색 자동 채색을 진행합니다.

그리고 밑색이 채색된 이미지를 S3에 업로드합니다.

현재 밑색 채색 기능은 아직 미완성이며

밑색 채색을 하기 전 영역 구분을 해주는 segmentation까지만 수행합니다.



## Install and Run

1. 먼저 프로그램을 실행하기 위해서 Python 3.8 환경이 필요합니다. 그리고 repository를 clone합니다.

```
$ git clone https://git.swmgit.org/swm-12/12_swm40/Null-AI.git
```

2. 의존성 라이브러리를 설치합니다.

```
$ pip install -r requirements.txt
```

3. `/web_app/config/s3_config.py`을 작성합니다

```
# s3_config.py
AWS_ACCESS_KEY = "S3 엑세스 키"
AWS_SECRET_ACCESS_KEY = "S3 시크릿 엑세스 키"
AWS_S3_BUCKET_REGION = "S3 리전 이름"
AWS_S3_BUCKET_NAME = "S3 버킷 이름"
```

4. 서버를 실행합니다.

```
$ cd web_app
$ python main.py
```



