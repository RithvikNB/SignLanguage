import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import google.protobuf.descriptor as d
import google.protobuf.symbol_database as s
import google.protobuf.message_factory as mf
if not hasattr(d.FieldDescriptor, "label"):
    d.FieldDescriptor.label = property(lambda self: self._label)
s.Default().GetPrototype = mf.GetMessageClass
import cv2
import mediapipe as mp

LIMIT = 800
SRC_DIR = os.path.join("asl_alphabet_train", "asl_alphabet_train")
DST_DIR = "data"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def crop_hand(image, landmarks, pad=0.2):
    h, w = image.shape[:2]
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0, int(x_min - pad * bw))
    y_min = max(0, int(y_min - pad * bh))
    x_max = min(w, int(x_max + pad * bw))
    y_max = min(h, int(y_max + pad * bh))
    if x_max <= x_min or y_max <= y_min:
        return None
    return image[y_min:y_max, x_min:x_max]

def process_dataset():
    os.makedirs(DST_DIR, exist_ok=True)
    classes = sorted(os.listdir(SRC_DIR))
    for cls_name in classes:
        src_cls_path = os.path.join(SRC_DIR, cls_name)
        if not os.path.isdir(src_cls_path):
            continue
        dst_cls_path = os.path.join(DST_DIR, cls_name)
        os.makedirs(dst_cls_path, exist_ok=True)
        images = sorted(os.listdir(src_cls_path))[:LIMIT]
        kept = 0
        for img_name in images:
            img_path = os.path.join(src_cls_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            if cls_name == "nothing":
                cropped = cv2.resize(image, (224, 224))
                cv2.imwrite(os.path.join(dst_cls_path, img_name), cropped)
                kept += 1
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    cropped = crop_hand(image, results.multi_hand_landmarks[0])
                    if cropped is not None and cropped.size > 0:
                        resized = cv2.resize(cropped, (224, 224))
                        cv2.imwrite(os.path.join(dst_cls_path, img_name), resized)
                        kept += 1
        print(f"{cls_name}: {kept}/{len(images)} images kept")
    hands.close()

if __name__ == "__main__":
    process_dataset()
