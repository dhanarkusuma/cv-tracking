import cv2
import numpy as np

# =====================================
# CONFIG
# =====================================

VIDEO_PATH  = "videos/static_object.mp4"
OUTPUT_PATH = "output/manual_lucas_kanade_static_object.mp4"

WINDOW_SIZE    = 10
MAX_ITERATIONS = 20
EPS            = 0.01
MAX_FLOW       = 30
MIN_FEATURES   = 8
REDETECT_EVERY = 20
PYRAMID_LEVELS = 3

# =====================================
# MANUAL LK - SINGLE POINT, ITERATIVE
# =====================================

def build_pyramid(img, levels):
    pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        blurred    = cv2.GaussianBlur(pyr[-1], (5, 5), 1.0)
        downsampled = blurred[::2, ::2]
        pyr.append(downsampled)
    return pyr

def manual_lk_point(prev_pyr, curr_pyr, px, py, levels, win):
    total_dx = 0.0
    total_dy = 0.0

    for level in range(levels - 1, -1, -1):
        scale  = 2 ** level
        prev_l = prev_pyr[level]
        curr_l = curr_pyr[level]
        h_l, w_l = prev_l.shape

        lpx = px / scale
        lpy = py / scale
        dx  = total_dx / scale
        dy  = total_dy / scale

        for _ in range(MAX_ITERATIONS):
            p0x = int(round(lpx))
            p0y = int(round(lpy))
            cx  = lpx + dx
            cy  = lpy + dy
            cxi = int(cx)
            cyi = int(cy)

            if (p0x < win or p0y < win or
                p0x >= w_l - win or p0y >= h_l - win):
                break
            if (cxi < win or cyi < win or
                cxi >= w_l - win - 1 or cyi >= h_l - win - 1):
                break

            prev_patch = prev_l[p0y-win:p0y+win+1, p0x-win:p0x+win+1]
            curr_patch = curr_l[cyi-win:cyi+win+1, cxi-win:cxi+win+1]

            if prev_patch.shape != curr_patch.shape:
                break

            Ix  = cv2.Sobel(prev_patch, cv2.CV_64F, 1, 0, ksize=3)
            Iy  = cv2.Sobel(prev_patch, cv2.CV_64F, 0, 1, ksize=3)
            It  = curr_patch - prev_patch

            Ixx = np.sum(Ix * Ix)
            Ixy = np.sum(Ix * Iy)
            Iyy = np.sum(Iy * Iy)
            det = Ixx * Iyy - Ixy ** 2

            if abs(det) < 1e-5:
                break

            ATb   = np.array([-np.sum(Ix * It), -np.sum(Iy * It)])
            ATA   = np.array([[Ixx, Ixy], [Ixy, Iyy]])
            delta = np.linalg.inv(ATA) @ ATb

            dx += delta[0]
            dy += delta[1]

            if np.hypot(delta[0], delta[1]) < EPS:
                break

        total_dx = dx * scale
        total_dy = dy * scale

    return total_dx, total_dy

# =====================================
# FORWARD-BACKWARD ERROR CHECK
# =====================================

def fb_error(prev_pyr, curr_pyr, px, py, dx, dy, levels, win):
    """
    Track forward px,py → new point, then track backward.
    Return the distance between original and back-tracked point.
    Small error = reliable track.
    """
    new_px = px + dx
    new_py = py + dy

    bdx, bdy = manual_lk_point(curr_pyr, prev_pyr, new_px, new_py, levels, win)

    back_px = new_px + bdx
    back_py = new_py + bdy

    return np.hypot(back_px - px, back_py - py)

FB_THRESHOLD = 2.0   # pixels — tracks with higher error are rejected

# =====================================
# FEATURE DETECTION — CENTER-WEIGHTED
# =====================================

def detect_features(gray, bx, by, bw, bh):
    """
    Detect features but DOWN-WEIGHT the border region of the box
    so we pick object features, not background edge features.
    """
    bx, by = int(bx), int(by)

    # Shrink the detection zone to the inner 70% of the box
    shrink_x = int(bw * 0.15)
    shrink_y = int(bh * 0.15)

    ix = bx + shrink_x
    iy = by + shrink_y
    iw = bw - 2 * shrink_x
    ih = bh - 2 * shrink_y

    # Guard against tiny box
    if iw < 10 or ih < 10:
        ix, iy, iw, ih = bx, by, bw, bh

    roi = gray[iy:iy+ih, ix:ix+iw]

    corners = cv2.goodFeaturesToTrack(
        roi,
        maxCorners=80,
        qualityLevel=0.02,    # slightly stricter quality
        minDistance=7,
        blockSize=7
    )

    if corners is None:
        return []

    pts = []
    for c in corners:
        fx, fy = c.ravel()
        pts.append([float(fx + ix), float(fy + iy)])

    return pts

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
# SELECT ROI
# =====================================

print("Select ROI and press ENTER")
bbox = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = bbox
box_x, box_y = float(x), float(y)

# =====================================
# INIT
# =====================================

prev_gray      = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
frame_h, frame_w = prev_gray.shape
prev_pyr       = build_pyramid(prev_gray, PYRAMID_LEVELS)
feature_points = detect_features(prev_gray, box_x, box_y, w, h)

if not feature_points:
    raise Exception("No features found in ROI")

# =====================================
# VIDEO WRITER
# =====================================

fps    = cap.get(cv2.CAP_PROP_FPS) or 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
if not out.isOpened():
    raise Exception("VideoWriter failed")

# =====================================
# TRACKING LOOP
# =====================================

frame_count           = 0
frames_since_redetect = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    curr_pyr = build_pyramid(gray, PYRAMID_LEVELS)
    output   = frame.copy()

    new_points = []
    dx_list    = []
    dy_list    = []

    # =================================
    # TRACK + FORWARD-BACKWARD CHECK
    # =================================

    for px, py in feature_points:

        dx, dy = manual_lk_point(
            prev_pyr, curr_pyr, px, py, PYRAMID_LEVELS, WINDOW_SIZE
        )

        # 1. Magnitude outlier rejection
        if abs(dx) > MAX_FLOW or abs(dy) > MAX_FLOW:
            continue

        # 2. Forward-backward consistency check
        err = fb_error(prev_pyr, curr_pyr, px, py, dx, dy, PYRAMID_LEVELS, WINDOW_SIZE)
        if err > FB_THRESHOLD:
            continue          # unreliable track — discard

        new_x = px + dx
        new_y = py + dy

        if (WINDOW_SIZE < new_x < frame_w - WINDOW_SIZE and
                WINDOW_SIZE < new_y < frame_h - WINDOW_SIZE):

            new_points.append([new_x, new_y])
            dx_list.append(dx)
            dy_list.append(dy)

            cv2.arrowedLine(output,
                            (int(px), int(py)), (int(new_x), int(new_y)),
                            (0, 255, 0), 2, tipLength=0.4)
            cv2.circle(output, (int(new_x), int(new_y)), 3, (0, 0, 255), -1)

    # =================================
    # ROBUST BOX UPDATE
    # =================================

    if len(dx_list) >= 2:
        avg_dx = np.median(dx_list)
        avg_dy = np.median(dy_list)
        box_x += avg_dx
        box_y += avg_dy

    box_x = float(np.clip(box_x, 0, frame_w - w))
    box_y = float(np.clip(box_y, 0, frame_h - h))
    bxi, byi = int(box_x), int(box_y)

    # =================================
    # CARRY POINTS FORWARD
    # =================================

    feature_points        = new_points
    frames_since_redetect += 1

    if (len(feature_points) < MIN_FEATURES or
            frames_since_redetect >= REDETECT_EVERY):
        feature_points        = detect_features(gray, box_x, box_y, w, h)
        frames_since_redetect = 0

    # =================================
    # DRAW
    # =================================

    cv2.rectangle(output, (bxi, byi), (bxi + w, byi + h), (255, 0, 0), 3)

    cv2.putText(output, "Manual Lucas-Kanade Tracking",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(output, f"Tracked Features: {len(feature_points)}",
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output, f"FB-Validated: {len(dx_list)}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Manual Lucas-Kanade Tracking", output)
    out.write(output)

    prev_pyr  = curr_pyr
    prev_gray = gray.copy()
    frame_count += 1

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# =====================================
# RELEASE
# =====================================

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nProcessed {frame_count} frames")
print(f"Saved to: {OUTPUT_PATH}")