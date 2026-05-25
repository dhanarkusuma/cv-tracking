import cv2
import numpy as np

# =====================================
# CONFIG
# =====================================

VIDEO_PATH = "videos/move_object_motor.mp4"
VIDEO_OUTPUT_PATH = "output/move_object_motor.mp4"

# =====================================
# OPEN VIDEO
# =====================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Cannot open video")

ret, first_frame = cap.read()
# first_frame = cv2.resize(
#     first_frame,
#     (640, 360)
# )

if not ret:
    raise Exception("Cannot read first frame")

# =====================================
# ROI SELECTOR GUI
# =====================================

print("Select object ROI and press ENTER")

bbox = cv2.selectROI(
    "Select Object",
    first_frame,
    fromCenter=False,
    showCrosshair=True
)

cv2.destroyAllWindows()

x, y, w, h = bbox

print("\n=== ROI RESULT ===")
print(f"x = {x}")
print(f"y = {y}")
print(f"w = {w}")
print(f"h = {h}")
print("\nTemplate Captured.")

# =====================================
# CREATE TEMPLATE
# =====================================

template = first_frame[y:y+h, x:x+w]

gray_template = cv2.cvtColor(
    template,
    cv2.COLOR_BGR2GRAY
)

# =====================================
# VIDEO WRITER
# =====================================

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_width = 640
# frame_height = 360

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    VIDEO_OUTPUT_PATH,
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    raise Exception("VideoWriter failed")

# =====================================
# RESET VIDEO TO FRAME 0
# =====================================

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.resize(
    #     frame,
    #     (640, 360)
    # )
    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    min_ssd = float("inf")

    best_x = 0
    best_y = 0

    # =================================
    # SLIDING WINDOW
    # =================================

    for j in range(0, gray_frame.shape[0] - h, 20):
        for i in range(0, gray_frame.shape[1] - w, 20):
            window = gray_frame[
                j:j+h,
                i:i+w
            ]

            # =========================
            # SSD CALCULATION
            # =========================

            ssd = np.sum(( window.astype(np.float32) - gray_template.astype(np.float32)) ** 2)

            # =========================
            # FIND BEST MATCH
            # =========================

            if ssd < min_ssd:
                min_ssd = ssd
                best_x = i
                best_y = j

    # =================================
    # DRAW RECTANGLE
    # =================================

    top_left = (best_x, best_y)

    bottom_right = (
        best_x + w,
        best_y + h
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        3
    )

    # =================================
    # TEXT
    # =================================

    cv2.putText(
        frame,
        "Manual Template Matching",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"SSD: {min_ssd:.0f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =================================
    # SAVE
    # =================================

    out.write(frame)

    frame_count += 1

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =====================================
# RELEASE
# =====================================

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count} frames")