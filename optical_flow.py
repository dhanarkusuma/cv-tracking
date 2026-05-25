import cv2
import numpy as np
import os

# =====================================
# CONFIG
# =====================================
VIDEO_PATH = "videos/move_object_motor.mp4"
VIDEO_OUTPUT_PATH = "output/move_object_motor_optical_flow.mp4"

MIN_POINTS = 5          # Re-detect if tracked points fall below this
TRAIL_LENGTH = 30       # How many frames of trail history to keep
TRAIL_FADE = True       # Whether to fade older trail segments

os.makedirs("output", exist_ok=True)


# =====================================
# FUNCTION: OPTICAL FLOW MANUAL (LK)
# =====================================
def optical_flow_manual(im1, im2, points, window_size=7):
    """
    Menghitung Optical Flow secara manual menggunakan metode Lucas-Kanade.
    """
    half_w = window_size // 2

    im1_blur = cv2.GaussianBlur(im1, (5, 5), 1.0)
    im2_blur = cv2.GaussianBlur(im2, (5, 5), 1.0)

    Ix = cv2.Sobel(im1_blur, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(im1_blur, cv2.CV_64F, 0, 1, ksize=3)
    It = im2_blur.astype(np.float64) - im1_blur.astype(np.float64)

    good_new_points = []
    status = []

    # FIX: normalise shape → always (N, 2)
    pts = points.reshape(-1, 2)

    for pt in pts:
        x_f, y_f = float(pt[0]), float(pt[1])
        x, y = int(round(x_f)), int(round(y_f))

        if (y - half_w < 0 or y + half_w >= im1.shape[0] or
                x - half_w < 0 or x + half_w >= im1.shape[1]):
            status.append(0)
            good_new_points.append([x_f, y_f])
            continue

        A_x = Ix[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()
        A_y = Iy[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()
        B   = -It[y - half_w: y + half_w + 1, x - half_w: x + half_w + 1].flatten()

        A = np.vstack((A_x, A_y)).T
        ATA = A.T @ A

        if abs(np.linalg.det(ATA)) < 1e-5:
            status.append(0)
            good_new_points.append([x_f, y_f])
        else:
            ATB = A.T @ B
            try:
                u, v = np.linalg.solve(ATA, ATB)
            except np.linalg.LinAlgError:
                status.append(0)
                good_new_points.append([x_f, y_f])
                continue

            if np.sqrt(u ** 2 + v ** 2) > 20:
                status.append(0)
                good_new_points.append([x_f, y_f])
            else:
                status.append(1)
                good_new_points.append([x_f + u, y_f + v])

    # Return shape (N, 1, 2) to stay consistent with OpenCV convention
    return (np.array(good_new_points, dtype=np.float32).reshape(-1, 1, 2),
            np.array(status, dtype=np.uint8))


# =====================================
# HELPER: Re-detect feature points inside bbox
# =====================================
def redetect_points(gray, x, y, w, h):
    mask_roi = np.zeros_like(gray)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(gray.shape[1], x + w)
    y2 = min(gray.shape[0], y + h)
    mask_roi[y1:y2, x1:x2] = 255
    pts = cv2.goodFeaturesToTrack(gray, mask=mask_roi,
                                  maxCorners=50, qualityLevel=0.3, minDistance=7)
    return pts


# =====================================
# OPEN VIDEO & SELECT ROI
# =====================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Tidak bisa membuka video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise IOError("Tidak bisa membaca frame pertama dari video!")

FRAME_HEIGHT, FRAME_WIDTH = first_frame.shape[:2]

print("Select ROI and press ENTER or SPACE. Press C to cancel.")
bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()
x, y, w, h = bbox

if w == 0 or h == 0:
    raise ValueError("ROI tidak valid (lebar atau tinggi = 0).")

old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = redetect_points(old_gray, x, y, w, h)

if p0 is None or len(p0) == 0:
    raise Exception("Tidak ada titik fitur terdeteksi di ROI!")

centroid_trail = []

# =====================================
# VIDEO WRITER
# =====================================
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

frame_count = 0

# =====================================
# LOOP TRACKING
# =====================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st = optical_flow_manual(old_gray, frame_gray, p0, window_size=7)

    # ─────────────────────────────────────────────────────────────────────
    # FIX: p1 shape is (N,1,2); boolean index gives (K,1,2) not (K,2).
    #      Reshape to (K,2) immediately so [:, 0] and [:, 1] always work.
    # ─────────────────────────────────────────────────────────────────────
    good_new = p1[st == 1].reshape(-1, 2)
    good_old = p0.reshape(-1, 2)[st == 1]

    # --- Re-detect jika poin terlalu sedikit ---
    if len(good_new) < MIN_POINTS:
        print(f"Frame {frame_count}: poin tersisa {len(good_new)}, re-deteksi...")
        p0_redet = redetect_points(frame_gray, x, y, w, h)
        if p0_redet is not None and len(p0_redet) >= MIN_POINTS:
            p0 = p0_redet
            old_gray = frame_gray.copy()
            frame_count += 1
            continue
        else:
            print("Re-deteksi gagal, menghentikan tracking.")
            break

    # --- Update bounding box dinamis ---
    # good_new is now safely (K, 2) → [:, 0] and [:, 1] are valid
    x_coords = good_new[:, 0]
    y_coords = good_new[:, 1]
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    padding = 15
    x = max(0, x_min - padding)
    y = max(0, y_min - padding)
    w = min(FRAME_WIDTH,  x_max + padding) - x
    h = min(FRAME_HEIGHT, y_max + padding) - y

    # --- Centroid ---
    cx = int(np.mean(x_coords))
    cy = int(np.mean(y_coords))
    centroid_trail.append((cx, cy))
    if len(centroid_trail) > TRAIL_LENGTH:
        centroid_trail.pop(0)

    # =====================================
    # GAMBAR TRAIL (motion history lines)
    # =====================================
    n = len(centroid_trail)
    for i in range(1, n):
        alpha = i / n                           # 0.0 (tertua) → 1.0 (terbaru)
        thickness = max(1, int(3 * alpha))
        if TRAIL_FADE:
            color = (0, int(255 * (1 - alpha)), 255)  # kuning → merah
            overlay = frame.copy()
            cv2.line(overlay, centroid_trail[i - 1], centroid_trail[i], color, thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            cv2.line(frame, centroid_trail[i - 1], centroid_trail[i], (0, 200, 255), 2)

    # --- Titik pelacak merah ---
    for new_pt in good_new:
        a, b = int(new_pt[0]), int(new_pt[1])
        cv2.circle(frame, (a, b), 4, (0, 0, 255), -1)

    # --- Bounding box hijau ---
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- Centroid biru ---
    cv2.circle(frame, (cx, cy), 6, (255, 100, 0), -1)

    # --- Info teks ---
    cv2.putText(frame, f"Tracked Points : {len(good_new)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame          : {frame_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Manual Optical Flow Tracking", frame)
    out.write(frame)

    # --- Persiapan frame berikutnya ---
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)   # store back as (N,1,2) for next iteration
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Dihentikan oleh pengguna.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Selesai! Output disimpan di: {VIDEO_OUTPUT_PATH}")
