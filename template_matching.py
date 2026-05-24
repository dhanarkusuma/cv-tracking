import cv2

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

# =====================================
# TRACKING LOOP
# =====================================

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # ================================
    # FRAME TO GRAYSCALE
    # ================================

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ================================
    # TEMPLATE MATCHING
    # ================================

    result = cv2.matchTemplate(
        gray_frame,
        gray_template,
        cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc

    bottom_right = (
        top_left[0] + w,
        top_left[1] + h
    )

    # ================================
    # DRAW RECTANGLE
    # ================================

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (0, 255, 0),
        3
    )

    # ================================
    # CONFIDENCE TEXT
    # ================================

    cv2.putText(
        frame,
        f"Confidence: {max_val:.2f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # ================================
    # METHOD LABEL
    # ================================

    cv2.putText(
        frame,
        "Template Matching",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # ================================
    # SAVE FRAME
    # ================================

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

print(f"\nFinished processing {frame_count} frames")