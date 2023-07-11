yolo = None
mediapipe = None
def setup_yolo():
    import YOLOv5
    global yolo
    yolo = YOLOv5
    return yolo

def setup_mediapipe():
    import MediaPipe
    global mediapipe
    mediapipe = MediaPipe
    return mediapipe


def has_hand(frame, thresh, model):
    global yolo
    global mediapipe
    if model == 'yolo':
        if yolo == None:
            yolo = setup_yolo()
            return yolo.yolo_model(frame, thresh)
        else:
            return yolo.yolo_model(frame, thresh)
    elif model == 'mediapipe':
        if mediapipe == None:
            mediapipe = setup_mediapipe()
            return mediapipe.mediapipe_model(frame)
        else:
            return mediapipe.mediapipe_model(frame)


frame = 'fileobjects/photos/man2.jpg'
print(has_hand(frame, 0.8, 'yolo'))