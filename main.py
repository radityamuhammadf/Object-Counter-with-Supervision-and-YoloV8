#library opencv
import cv2
#library argument parser
import argparse
#library YOLO --> object detection
from ultralytics import YOLO

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
    cap=cv2.VideoCapture(0)
    #deklarasi resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    #deklarasi model YOLO yang akan digunakan (YOLOv8)
    model=YOLO("yolov8l.pt")

    # cek kondisi variabel capture
    while True:
        #ambil data frame dan ret dari videocapture utk ditampilkan 
        ret, frame = cap.read()
        #tampilkan hasil detection yolov8
        result=model(frame)
        #tampilkan ke gui
        cv2.imshow("yolov8",frame)

        #trigger untuk exit
        if(cv2.waitKey(30)==27):
            break

if __name__ == "__main__":
    main()