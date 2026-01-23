"""
세그멘테이션 기반 개인형 이동장치 불법 주차 단속 시스템
Segmentation-based Illegal parking enforcement system for Personal-mobility

Based on: YOLOv5 Instance Segmentation
Authors: 이현우, 이소연, 김민승, 김예서, 김대영 (순천향대학교)

이 코드는 YOLOv5 segment/predict.py를 기반으로 수정되었습니다.
"""

import argparse
import os
import sys
from pathlib import Path
import math

import torch
import numpy as np
import cv2

# YOLOv5 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_boxes,
    scale_segments,
)
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages, LoadStreams


class IllegalParkingDetector:
    """
    점자 블록 위 개인형 이동장치 불법 주차 탐지 시스템
    
    논문의 시스템 구조:
    1. Detection Engine: 개인형 이동장치와 점자 블록 인식
    2. Checking Engine: 불법 주차 상황 판단 로직
    """
    
    # 클래스 라벨 정의
    CLASS_GUIDE_BLOCK = 0  # 점자 블록
    CLASS_SCOOTER = 1      # 개인형 이동장치 (킥보드)
    
    def __init__(
        self,
        weights='best.pt',           # 학습된 모델 가중치
        device='',                   # cuda device or cpu
        imgsz=(640, 640),           # 입력 이미지 크기
        conf_thres=0.25,            # confidence threshold
        iou_thres=0.45,             # NMS IOU threshold
        illegal_distance=100,        # 불법 주차 판단 거리 임계값 (픽셀)
        half=False,                  # FP16 half-precision inference
    ):
        """
        Args:
            weights: 학습된 YOLOv5 세그멘테이션 모델 경로
            device: 추론 디바이스
            imgsz: 입력 이미지 크기
            conf_thres: confidence threshold
            iou_thres: NMS IOU threshold  
            illegal_distance: 불법 주차 판단 거리 임계값 (픽셀 단위)
            half: FP16 사용 여부
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.illegal_distance = illegal_distance
        self.imgsz = imgsz
        
        # 디바이스 선택
        self.device = select_device(device)
        
        # 모델 로드
        self.model = DetectMultiBackend(weights, device=self.device, fp16=half)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        
        # 이미지 크기 체크
        self.imgsz = check_img_size(imgsz, s=self.stride)
        
        # 모델 warmup
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        
        print(f"모델 로드 완료: {weights}")
        print(f"클래스: {self.names}")
        print(f"불법 주차 판단 거리: {self.illegal_distance} 픽셀")

    def calculate_center(self, xyxy):
        """
        Bounding box의 중심점 계산
        
        논문 Algorithm 1, line 3-4:
        center_x ← int((xyxy[0] + xyxy[2])/2)
        center_y ← int((xyxy[1] + xyxy[3])/2)
        
        Args:
            xyxy: [x1, y1, x2, y2] 형태의 bounding box 좌표
            
        Returns:
            (center_x, center_y): 중심점 좌표
        """
        center_x = int((xyxy[0] + xyxy[2]) / 2)
        center_y = int((xyxy[1] + xyxy[3]) / 2)
        return (center_x, center_y)

    def calculate_euclidean_distance(self, point1, point2):
        """
        유클라디안 거리 계산
        
        논문 Algorithm 1, line 13-15:
        x1 ← (guide_block_center[0] - scooter_center[0])²
        y1 ← (guide_block_center[1] - scooter_center[1])²
        line_length ← √(x1 + y1)
        
        Args:
            point1: (x1, y1) 첫 번째 점
            point2: (x2, y2) 두 번째 점
            
        Returns:
            float: 두 점 사이의 유클라디안 거리
        """
        x1 = (point1[0] - point2[0]) ** 2
        y1 = (point1[1] - point2[1]) ** 2
        distance = math.sqrt(x1 + y1)
        return distance

    def check_illegal_parking(self, detections):
        """
        불법 주차 판단 로직
        
        논문 Algorithm 1 구현:
        - 각 객체의 중심점 계산
        - 점자 블록과 이동장치 간 거리 측정
        - 임계값 이하면 불법 주차로 판단
        
        Args:
            detections: 탐지된 객체들의 정보 리스트
                       [(xyxy, class_id, confidence, mask), ...]
                       
        Returns:
            dict: {
                'is_illegal': bool,
                'guide_block_centers': list,
                'scooter_centers': list,
                'illegal_pairs': list of (scooter_center, block_center, distance)
            }
        """
        guide_block_centers = []
        scooter_centers = []
        
        # Algorithm 1, line 1-10: 각 객체별 중심점 분류
        for det in detections:
            xyxy, cls, conf, mask = det
            center = self.calculate_center(xyxy)
            
            # line 5-9: 클래스별 중심점 리스트에 저장
            if cls == self.CLASS_GUIDE_BLOCK:  # 점자 블록
                guide_block_centers.append(center)
            elif cls == self.CLASS_SCOOTER:    # 개인형 이동장치
                scooter_centers.append(center)
        
        # Algorithm 1, line 11-20: 모든 조합에 대해 거리 계산 및 판단
        illegal_pairs = []
        is_illegal = False
        
        for scooter_center in scooter_centers:
            for block_center in guide_block_centers:
                # line 13-15: 유클라디안 거리 계산
                distance = self.calculate_euclidean_distance(
                    block_center, scooter_center
                )
                
                # line 16-20: 불법 주차 판단
                if distance <= self.illegal_distance:
                    is_illegal = True
                    illegal_pairs.append((scooter_center, block_center, distance))
                    print(f"불법 주차 탐지! 거리: {distance:.2f} 픽셀")
        
        return {
            'is_illegal': is_illegal,
            'guide_block_centers': guide_block_centers,
            'scooter_centers': scooter_centers,
            'illegal_pairs': illegal_pairs
        }

    def preprocess(self, img):
        """이미지 전처리"""
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]  # batch 차원 추가
        return img

    def predict(self, source, save_dir='runs/detect', view_img=False, save_img=True):
        """
        이미지/비디오/스트림에서 불법 주차 탐지 수행
        
        Args:
            source: 입력 소스 (이미지 경로, 비디오 경로, 웹캠 번호, RTSP URL 등)
            save_dir: 결과 저장 디렉토리
            view_img: 실시간 결과 표시 여부
            save_img: 결과 이미지 저장 여부
            
        Returns:
            list: 각 프레임별 탐지 결과
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 로더 설정
        is_stream = source.isnumeric() or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://')
        )
        
        if is_stream:
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        
        results = []
        
        for path, im, im0s, vid_cap, s in dataset:
            # 전처리
            im = self.preprocess(im)
            
            # 추론
            pred, proto = self.model(im)[:2]
            
            # NMS
            pred = non_max_suppression(
                pred, 
                self.conf_thres, 
                self.iou_thres, 
                nm=32  # mask 채널 수
            )
            
            # 결과 처리
            for i, det in enumerate(pred):
                if is_stream:
                    p, im0 = path[i], im0s[i].copy()
                else:
                    p, im0 = path, im0s.copy()
                
                p = Path(p)
                save_path = str(save_dir / p.name)
                
                annotator = Annotator(im0, line_width=2)
                
                detections = []
                
                if len(det):
                    # Mask 처리
                    masks = process_mask(
                        proto[i], 
                        det[:, 6:], 
                        det[:, :4], 
                        im.shape[2:], 
                        upsample=True
                    )
                    
                    # 좌표 스케일링
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    # 각 탐지 결과 처리
                    for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):
                        cls = int(cls)
                        mask = masks[j].cpu().numpy()
                        
                        detections.append((
                            [int(x) for x in xyxy],
                            cls,
                            float(conf),
                            mask
                        ))
                        
                        # 시각화
                        label = f'{self.names[cls]} {conf:.2f}'
                        color = colors(cls, True)
                        annotator.box_label(xyxy, label, color=color)
                        
                        # 마스크 시각화
                        mask_colored = np.zeros_like(im0)
                        mask_colored[mask > 0.5] = color
                        im0 = cv2.addWeighted(im0, 1, mask_colored, 0.3, 0)
                
                # 불법 주차 판단
                parking_result = self.check_illegal_parking(detections)
                
                # 결과 시각화
                im0 = self.visualize_result(im0, parking_result)
                
                results.append({
                    'path': str(p),
                    'detections': detections,
                    'parking_result': parking_result
                })
                
                # 결과 출력
                status = "불법 주차!" if parking_result['is_illegal'] else "정상"
                print(f"{p.name}: {status}")
                
                # 이미지 표시
                if view_img:
                    cv2.imshow('Illegal Parking Detection', im0)
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                # 이미지 저장
                if save_img:
                    cv2.imwrite(save_path, im0)
                    print(f"결과 저장: {save_path}")
        
        if view_img:
            cv2.destroyAllWindows()
        
        return results

    def visualize_result(self, img, parking_result):
        """
        불법 주차 판단 결과 시각화
        
        Args:
            img: 원본 이미지
            parking_result: check_illegal_parking() 반환값
            
        Returns:
            시각화된 이미지
        """
        img = img.copy()
        
        # 점자 블록 중심점 표시 (초록색)
        for center in parking_result['guide_block_centers']:
            cv2.circle(img, center, 8, (0, 255, 0), -1)
            cv2.putText(img, 'Block', (center[0]-20, center[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 이동장치 중심점 표시 (파란색)
        for center in parking_result['scooter_centers']:
            cv2.circle(img, center, 8, (255, 0, 0), -1)
            cv2.putText(img, 'Scooter', (center[0]-20, center[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 불법 주차 쌍 연결선 및 경고 표시
        for scooter_center, block_center, distance in parking_result['illegal_pairs']:
            # 빨간 연결선
            cv2.line(img, scooter_center, block_center, (0, 0, 255), 3)
            
            # 거리 표시
            mid_point = (
                (scooter_center[0] + block_center[0]) // 2,
                (scooter_center[1] + block_center[1]) // 2
            )
            cv2.putText(img, f'{distance:.1f}px', mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 상단 상태 표시
        if parking_result['is_illegal']:
            cv2.rectangle(img, (0, 0), (300, 50), (0, 0, 255), -1)
            cv2.putText(img, 'ILLEGAL PARKING!', (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(img, (0, 0), (200, 50), (0, 255, 0), -1)
            cv2.putText(img, 'Legal Parking', (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return img


def run(
    weights='best.pt',
    source='test.jpg',
    device='',
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    illegal_distance=100,
    save_dir='runs/detect',
    view_img=False,
    no_save=False,
    half=False,
):
    """
    메인 실행 함수
    """
    detector = IllegalParkingDetector(
        weights=weights,
        device=device,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        illegal_distance=illegal_distance,
        half=half,
    )
    
    results = detector.predict(
        source=source,
        save_dir=save_dir,
        view_img=view_img,
        save_img=not no_save,
    )
    
    # 최종 통계 출력
    total = len(results)
    illegal_count = sum(1 for r in results if r['parking_result']['is_illegal'])
    print(f"\n=== 탐지 완료 ===")
    print(f"총 프레임/이미지: {total}")
    print(f"불법 주차 탐지: {illegal_count}")
    print(f"정상: {total - illegal_count}")
    
    return results


def parse_opt():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='점자 블록 위 개인형 이동장치 불법 주차 탐지 시스템'
    )
    parser.add_argument('--weights', type=str, default='best.pt',
                       help='학습된 모델 가중치 경로')
    parser.add_argument('--source', type=str, default='test.jpg',
                       help='입력 소스 (이미지/비디오/웹캠/스트림)')
    parser.add_argument('--device', default='',
                       help='cuda device (예: 0 또는 0,1,2,3 또는 cpu)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, 
                       default=[640], help='추론 이미지 크기')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='NMS IOU threshold')
    parser.add_argument('--illegal-distance', type=float, default=100,
                       help='불법 주차 판단 거리 임계값 (픽셀)')
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                       help='결과 저장 디렉토리')
    parser.add_argument('--view-img', action='store_true',
                       help='결과 실시간 표시')
    parser.add_argument('--no-save', action='store_true',
                       help='결과 저장 안함')
    parser.add_argument('--half', action='store_true',
                       help='FP16 half-precision 사용')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main():
    """메인 함수"""
    opt = parse_opt()
    run(**vars(opt))


if __name__ == '__main__':
    main()
