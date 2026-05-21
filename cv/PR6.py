import cv2, numpy as np

cap = cv2.VideoCapture(0)
N = 800

ret, frame = cap.read()
x, y, w, h = cv2.selectROI(frame, False)

roi = frame[y:y+h, x:x+w]
ref_color = roi.mean(axis=(0,1))

particles = np.tile([x+w/2, y+h/2], (N,1))
weights = np.ones(N) / N

while True:
    ret, frame = cap.read()
    if not ret: break

    h_img, w_img = frame.shape[:2]

    # Predict
    particles += np.random.randn(N,2) * 15

    # Keep particles inside frame
    particles[:,0] = np.clip(particles[:,0], 0, w_img-1)
    particles[:,1] = np.clip(particles[:,1], 0, h_img-1)

    # Measurement (vectorized)
    coords = particles.astype(int)
    colors = frame[coords[:,1], coords[:,0]]
    dist = np.linalg.norm(colors - ref_color, axis=1)
    weights = 1 / (dist + 1)

    weights /= weights.sum()

    # Estimate
    pos = np.average(particles, weights=weights, axis=0).astype(int)

    # Resample
    idx = np.random.choice(N, N, p=weights)
    particles = particles[idx]
    weights.fill(1.0 / N)

    # Draw
    for p in particles:
        cv2.circle(frame, tuple(p.astype(int)), 1, (0,255,0), -1)
    cv2.circle(frame, tuple(pos), 8, (0,0,255), -1)

    cv2.imshow("Particle Filter", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()