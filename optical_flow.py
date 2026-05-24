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

ret, old_frame = cap.read()

if not ret:
    raise Exception("Cannot read first frame")

# =====================================
# SELECT ROI
# =====================================

print("Select object ROI and press ENTER")

bbox = cv2.selectROI(
    "Select Object",
    old_frame,
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
# CREATE ROI MASK
# =====================================

roi_mask = np.zeros_like(
    cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
)

roi_mask[y:y+h, x:x+w] = 255

# =====================================
# PREPARE FIRST FRAME
# =====================================

old_gray = cv2.cvtColor(
    old_frame,
    cv2.COLOR_BGR2GRAY
)

# =====================================
# DETECT FEATURES INSIDE ROI
# =====================================

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=roi_mask,
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

if p0 is None:
    raise Exception("No features detected")

# =====================================
# LUCAS-KANADE PARAMETERS
# =====================================

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
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
# MASK FOR DRAWING TRAJECTORY
# =====================================

draw_mask = np.zeros_like(old_frame)

# =====================================
# TRACKING LOOP
# =====================================

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =================================
    # OPTICAL FLOW CALCULATION
    # =================================

    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray,
        frame_gray,
        p0,
        None,
        **lk_params
    )

    if p1 is None:
        break

    # =================================
    # SELECT GOOD POINTS
    # =================================

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # =================================
    # DRAW TRACKING
    # =================================

    for new, old in zip(good_new, good_old):

        a, b = new.ravel()
        c, d = old.ravel()

        a, b, c, d = map(
            int,
            [a, b, c, d]
        )

        # Draw motion line
        draw_mask = cv2.line(
            draw_mask,
            (a, b),
            (c, d),
            (0, 255, 0),
            2
        )

        # Draw feature point
        frame = cv2.circle(
            frame,
            (a, b),
            5,
            (0, 0, 255),
            -1
        )

    # Combine frame + tracking lines
    output = cv2.add(frame, draw_mask)

    # =================================
    # DRAW ROI BOX
    # =================================

    cv2.rectangle(
        output,
        (x, y),
        (x+w, y+h),
        (255, 0, 0),
        2
    )

    # =================================
    # LABEL
    # =================================

    cv2.putText(
        output,
        "Optical Flow Tracking",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    #
    # # =================================
    # # SHOW FRAME
    # # =================================
    #
    # cv2.imshow(
    #     "Optical Flow",
    #     output
    # )

    # =================================
    # SAVE FRAME
    # =================================

    out.write(output)

    # =================================
    # UPDATE PREVIOUS FRAME
    # =================================

    old_gray = frame_gray.copy()

    p0 = good_new.reshape(
        -1,
        1,
        2
    )

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