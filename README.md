# ColorAid AI Server
소프트웨어 마에스트로 12기 NULL팀의 웹툰 자동채색 어플리케이션 ColorAid의 AI Server입니다.

선화 이미지의 S3 URL을 요청으로 받으면 그 이미지를 받아서 밑색 자동 채색을 진행합니다.

그리고 밑색이 채색된 이미지를 S3에 업로드합니다.

## Install and Run

1. 먼저 프로그램을 실행하기 위해서 **Python 3.6** 환경이 필요합니다. 그리고 repository를 clone합니다.

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

## How To Use

다음과 같이 HTTP 요청을 합니다.

```
curl --location --request POST 'http://localhost:8000/paint' \
--header 'Content-Type: application/json' \
--data-raw '{
    "referenceAccessKey": "레퍼런스 이미지 AccessKey",
    "sketchAccessKey": "스케치 이미지 AccessKey",
    "resultAccessKey": "결과 이미지 AccessKey"
}'
```

## Examples

| 레퍼런스 이미지                                              | 스케치 이미지                                                | 결과 이미지                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![06_color](https://user-images.githubusercontent.com/52124204/140652910-9f8ba7c1-bd0a-4c6f-a24b-843de3b43c5b.png) | ![06_sketch](https://user-images.githubusercontent.com/52124204/140652918-8a5c9a5a-3e1b-46a6-8933-6a1966c6b6cc.png) | ![06_sketch](https://user-images.githubusercontent.com/52124204/140652933-6b513090-cbfc-4b0b-a698-fd2049f7539b.png) |
| ![09_color](https://user-images.githubusercontent.com/52124204/140652924-edf34203-b51c-4fa0-b5c8-59b3ec9d6453.png) | ![09_sketch](https://user-images.githubusercontent.com/52124204/140652931-fd168fef-2d44-4fdb-b048-31087804b679.png) | ![09_sketch](https://user-images.githubusercontent.com/52124204/140652950-c6898aa8-4a48-4f95-b7f8-4d2286938a4e.png) |

