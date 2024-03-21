import cv2
import os

def record_video():
    # 設定影像儲存路徑
    save_path = r'C:\Users\jacky\Videos\cam\frames'
    video_save_path = r'C:\Users\jacky\Videos\cam\output.avi'

    # 創建儲存影像的資料夾
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 打開相機
    cap = cv2.VideoCapture(0)

    # 檢查相機是否成功打開
    if not cap.isOpened():
        print("無法打開相機")
        return

    # 設定視訊寬度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 設定視訊寫入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path, fourcc, 20.0, (width, height))

    frame_count = 0

    while cap.isOpened():
        # 從相機中讀取一帧影像
        ret, frame = cap.read()

        if ret:
            # 顯示影像
            cv2.imshow('frame', frame)

            # 寫入影像到 VideoWriter 對象
            out.write(frame)

            # 將每一幀影像保存到資料夾中
            cv2.imwrite(os.path.join(save_path, f'frame_{frame_count}.jpg'), frame)

            frame_count += 1

            # 按下 'q' 鍵退出迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_video()
