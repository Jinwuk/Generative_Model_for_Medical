## About Project

Pneumonia Treatment Progress Prediction using Diffusion Model 

<br><br><br>

## Installation 
- YAML 파일을 참고로 하여 필요한 패키지를 하나씩 설치한다.
- 중요한 것은 CUDA와 CUDNN, CUDA_Toolkit 등은 환경에 따라 다르므로 함부로 설치하게 되면 CUDA가 꼬여서 동작이 불가능해진다. 
- opencv 설치문제 때문에 Linux에서만 설치가 가능하다. Windows는 2024년 2월 현재, OpenCV의 버전이 낮아서 올바른 동작을 보장할 수 없다. (OpenCV 4.x 대 필요)

## Information of components in YAML File 
### args1.yaml
img_size:
  - 256
  - 256
Batch_Size: 2
EPOCHS: 100
T: 1000 			Lambda 적용 시간  - alpha 와 관련된 부분 (alpha/T)  
base_channels: 128		UNET Parameter		
beta_schedule: linear	    UNET Parameter 
channel_mults: ""		  UNET Parameter		
loss-type: l2			  UNET Parameter			
loss_weight: none		 UNET Parameter		
train_start: true		     No change : Training is default
lr: 1e-4                   
sample_distance: 600	Noise를 적용하는 부분 실제 T와 비교하여 작은 값을 사용 
weight_decay: 0.0
save_imgs: true
dropout: 0
attention_resolutions: 32,16,8    UNET parameter
num_heads: 2
num_head_channels: -1
noise_fn: gauss
dataset: custom				Maybe Kaggle 
device: cuda:0

##  File Description 

### NAS PT 파일 모음 
#### 위치
~~~
Document/003_AI_work/2024_medical_work/pt_files
~~~

#### Diffusion Model용 PT 파일
위에서 test_2024-0229 폴더내 파일들이다.
이중 2024_0229_PT 파일이 해당 
- **cls_pt** 폴더 : Classification 혹은 CAM용 PT 파일 
- **model** 폴더 : diff-params-ARGS=1 and diff-params-ARGS=2 폴더 아래에 UNET용 PT 파일 위치
- **_model** 폴더 : 위 폴더 내용의 backup 위치

### 입력시 모델 파일 (.PT) 이 위치하는 곳
~~~
model/diff-params-ARGS=1/params-final.pt
~~~
의 형식으로 위치한다. 

## Usage

examples
<br><br>

### Training

```
python diffusion_training.py -y <num> -i <image folder path> -d <device> -r <resume option>
```

- 모델의 파라미터 관련 옵션은 test_args 폴더에 args1.yaml 과 같이 yaml 파일로 저장하여 사용

- -y  :  args\<num>.yaml 의 숫자를 입력

- -i  :  학습용 이미지 폴더의 경로 (이 프로젝트에서는 학습시 정상 폐의 방사선 이미지만 사용)

- -d  :  사용할 gpu 설정 (기본값 = cuda:0)

- -r  :  "auto" 입력하여 사용하는 yaml로 학습한 모델이 있다면 이어서 학습 또는 pt file 의 경로를 입력하여 이어서 학습

- 예시

  - ```
    python diffusion_training.py -y 1 -i ../data/chest_xray/train/NORMAL/
    ```

<br><br>   

## Generate image

```
python generate_images.py -y <num> -i <image folder path> -d <device> -l <lambda list> -p <pt file path> -m <model name> --use_control_matrix
```

- 모델의 파라미터 관련 옵션은 test_args 폴더에 args1.yaml 과 같이 yaml 파일로 저장하여 사용

- -y  :  args\<num>.yaml 의 숫자를 입력

- -i  :  사용할 이미지 폴더의 경로

- -d  :  사용할 gpu 설정 (기본값 = cuda:0)

- -l  :  사용할 λ 값(노이즈 스텝) 입력

  - 입력예시

    - ```
      -l 100
      ```

    - ```
      -l 100,200,300
      ```

      __!! 여러 값 입력시 띄어쓰기 x !!__

- --use_control_matrix  :  제어행렬 사용 , -m, -p 파라미터 입력 필수

  - -p  :  CAM 값 구하는 데 사용할 pt 파일 경로
  - -m  :  모델 이름
    - -m 옵션으로 들어가는 모델명 목록
      - resnet50
      - resnet101
      - resnet152
      - densenet121
      - densenet201
      - densenet121_2048
        - fc layer 길이를 2048 로 늘린 모델

- 예시

  - ```
    python generate_images.py -y 1 -i ../data/chest_xray/test/PNEUMONIA/ -l 200,300,400
    ```
    
  - ```
    python generate_images.py -y 1 -i ../data/chest_xray/test/PNEUMONIA/ -l 200,300,400 -p ./resnet152.pt -m resnet152 --use_control_matrix
    ```

<br><br>



## 생성결과 정리용 코드 업데이트 예정 ..




<br><br>


### Reference

```
https://github.com/Julian-Wyatt/AnoDDPM
```
