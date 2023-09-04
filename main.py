#library opencv
import cv2
#library argument parser
import argparse
#library YOLO --> object detection
from ultralytics import YOLO
#library Supervision --> to make CV more simple
import supervision as sv
#numpy --> define polygon matrix
import numpy as np

# Full Size Screen Zone
# ZONE_PARAMS=np.array([
#     [0,0], #koordinat 1 -- kiri bawah
#     [1280,0], #koordinat 2 -- kanan bawah
#     [1280,720], #koordinat 3 -- kanan atas
#     [0,720] #koordinat 4 -- kiri atas 
# ])

#Constrained Zone
ZONE_PARAMS=np.array([
    [1280//2,0], #koordinat 1 -- kiri bawah
    [1280,0], #koordinat 2 -- kanan bawah
    [1280,720], #koordinat 3 -- kanan atas
    [1280//2,720] #koordinat 4 -- kiri atas 
])

#fungsi set resolusi
def parse_arguments()->argparse.Namespace:
    parser=argparse.ArgumentParser(description="YOLOv8 Live Detection")
    parser.add_argument(
        "--webcam-resolution", #nama argumennya
        default=[1280,720], #value defaultnya
        nargs=2, #jumlah argumen
        type=int #tipe data argumen
    )
    args=parser.parse_args() #parsing argument
    return args
    

# main program
def main():
    #instansiasi fungsi set resolusi
    args=parse_arguments()
    #ambil data
    frame_width,frame_height = args.webcam_resolution
    # inisiasi kamera --> argumennya buat pilih device
    cap=cv2.VideoCapture(1)
    #deklarasi resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    #deklarasi model YOLO yang akan digunakan (YOLOv8)
    model=YOLO("yolov8l.pt")

   #instansisasi boundingbox deteksi objek yolov8
    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    ) 

    #Instasisasi class PolygonZone (Supervision) untuk threshold
    zone=sv.PolygonZone(polygon=ZONE_PARAMS,frame_resolution_wh=tuple(args.webcam_resolution))
    #Gambarkan zona yang sebelumnya telah diinstasiasi di layar
    zone_annotator=sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=2,
        text_scale=2
    )


    # cek kondisi variabel capture
    while True:
        #ambil data frame dan ret dari videocapture utk ditampilkan 
        ret, frame = cap.read()
        #tampilkan elemen pertama dari hasil detection yolov8
        result=model(frame)[0]
        
        #melalui supervision, deteksi objek dengan YOLOv8
        detections=sv.Detections.from_yolov8(result)
        #filter yang dideteksi persons aja
        detections=detections[detections.class_id==0] 
        #panggil objek bounding box untuk nempel di hasil deteksi pada gambar (source: capture)
        
        labels=[
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _,confidence,class_id,_
            in detections
        ]
        #instansiasi objek labels untuk klasifikasi
        frame=box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        
        #mengaktifkan zona pada gambar yang diambil pada variabel detections
        zone.trigger(detections=detections)
        #menggambarkan zona pada frame yang dibuat
        frame=zone_annotator.annotate(scene=frame)

        #tampilkan ke gui
        cv2.imshow("yolov8",frame)

        #trigger untuk exit
        if(cv2.waitKey(30)==27):
            break

if __name__ == "__main__":
    main()