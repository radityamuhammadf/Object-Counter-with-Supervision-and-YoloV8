#library opencv
import cv2
#library argument parser
import argparse
#library YOLO --> object detection
from ultralytics import YOLO
#library Supervision --> to make CV more simple
import supervision as sv


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

    # cek kondisi variabel capture
    while True:
        #ambil data frame dan ret dari videocapture utk ditampilkan 
        ret, frame = cap.read()
        #tampilkan elemen pertama dari hasil detection yolov8
        result=model(frame)[0]
        
        #melalui supervision, deteksi objek dengan YOLOv8
        detections=sv.Detections.from_yolov8(result)
        #panggil objek bounding box untuk nempel di hasil deteksi pada gambar (source: capture)
        
        labels=[
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _,confidence,class_id,_
            in detections
        ]
        frame=box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        #instansiasi objek labels untuk 

        #tampilkan ke gui
        cv2.imshow("yolov8",frame)

        #trigger untuk exit
        if(cv2.waitKey(30)==27):
            break

if __name__ == "__main__":
    main()