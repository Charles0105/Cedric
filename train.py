from ultralytics import YOLO

def main():
    model = YOLO("RPN_YOLO.yaml")
    model.train(
        data="./TT100K/data.yaml",
        imgsz=640,
        epochs=500,
        patience=30, 
        device="0,1,2,3,4,5",
        batch=48, 
        workers=16,
    )

if __name__ == "__main__":
    main()