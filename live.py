import os, cv2, mediapipe as mp, numpy as np, tensorflow as tf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import google.protobuf.descriptor as d, google.protobuf.symbol_database as s, google.protobuf.message_factory as mf
if not hasattr(d.FieldDescriptor, "label"):
    d.FieldDescriptor.label = property(lambda self: self._label)
s.Default().GetPrototype = mf.GetMessageClass

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f if line.strip()]

model = tf.keras.models.load_model("asl_mobilenetv2.keras")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

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
        return None, None
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

cap = cv2.VideoCapture(0)
buffer, last_pred, streak = "", None, 0
current_label, current_conf = "None", 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        cropped, bbox = crop_hand(frame, results.multi_hand_landmarks[0])
        if cropped is not None and cropped.size > 0:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            rgb_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            inp = np.expand_dims(cv2.resize(rgb_crop, (224, 224)), axis=0)
            preds = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(preds))
            current_conf = float(preds[idx])
            current_label = labels[idx]
            if current_conf > 0.8:
                streak = streak + 1 if current_label == last_pred else 1
                last_pred = current_label
                if streak == 10:
                    if current_label == "space":
                        buffer += " "
                    elif current_label == "del":
                        buffer = buffer[:-1]
                    elif current_label != "nothing":
                        buffer += current_label
            else:
                last_pred, streak = None, 0
    else:
        current_label, current_conf, last_pred, streak = "None", 0.0, None, 0

    cv2.putText(frame, f"Pred: {current_label} ({current_conf:.2f})", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Buffer: {buffer}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("ASL Alphabet Translator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        buffer = ""

cap.release()
cv2.destroyAllWindows()
hands.close()
