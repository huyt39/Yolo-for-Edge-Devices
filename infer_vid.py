import cv2
from ultralytics import YOLO


model_path = "/home/labcv/huytuan/runs/detect/train6/weights/best.pt"  
video_path = "/mnt/e/workspace/Dataset/P-DESTR/dataset/P-DESTRE/videos/08-11-2019-1-1.MP4"                    
output_path = "output_video.mp4"              

# Load model
model = YOLO(model_path).to('cuda')

# Load video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, conf = 0.7)

   
    annotated_frame = results[0].plot()

  
    #cv2.imshow("Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
