import cv2
import os

def record_video():
    # 設定影像儲存路徑
    

    # 打開相機
    cap = cv2.VideoCapture(0)

    # 檢查相機是否成功打開
    if not cap.isOpened():
        print("無法打開相機")
        return

    # 設定視訊寬度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0

    while cap.isOpened():
        # 從相機中讀取一帧影像
        ret, frame = cap.read()

        if ret:
            # 顯示影像
            cv2.imshow('frame', frame)

            frame_count += 1

            # 按下 'q' 鍵退出迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 釋放資源
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

record_video()
