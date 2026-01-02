import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# 1. TUNING KNOBS
# ==========================================
BEND_SENSITIVITY = 0.4   # 1.0 = Strict Physics, 0.4 = Easier bending
MAX_VISUAL_BEND = 150    # Max pixels beam can drop
HOVER_THRESHOLD = -15.0  # Ignore finger if it is more than 15px ABOVE the line

# ==========================================
# 2. SETUP
# ==========================================
CAM_INDEX = 0  
cap = cv2.VideoCapture(CAM_INDEX)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# Green Tape Color
LOWER_BEAM = np.array([35, 80, 80], dtype=np.uint8)
UPPER_BEAM = np.array([85, 255, 255], dtype=np.uint8)

# Thresholds (Pixels)
NEAR_DIST = 120.0   
TOUCH_DIST = 50.0   

# Sim Panel
SIM_W, SIM_H = 600, 480
WALL_X = SIM_W - 50     
BEAM_Y = 240            
SIM_BEAM_LEN = 400      

# State
locked = False
ref_fixed = None     
ref_free = None      
ref_thickness = 10   

def get_cantilever_curve(p_start, p_end, num_points=20, deflection=0):
    points = []
    beam_vec = p_end - p_start
    length = np.linalg.norm(beam_vec)
    if length < 1e-3: return np.array([p_start])
    
    unit_beam = beam_vec / length
    # Standard Perpendicular Vector
    unit_perp = np.array([-unit_beam[1], unit_beam[0]]) 
    
    # CRITICAL: Ensure this vector points DOWN (Screen Y+)
    if unit_perp[1] < 0: unit_perp = -unit_perp

    for i in range(num_points + 1):
        t = i / num_points 
        base_pos = p_start + beam_vec * t
        displacement = (t ** 2) * abs(deflection)
        final_pos = base_pos + (unit_perp * displacement)
        points.append(final_pos)
    return np.array(points, dtype=np.int32)

print("Controls: 'c' Lock, 'u' Unlock, 'q' Quit")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    display = cv2.resize(frame, (640, 480))
    h, w, _ = display.shape
    hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)

    # A. HAND
    fx, fy = None, None
    img_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        tip = results.multi_hand_landmarks[0].landmark[8]
        fx, fy = int(tip.x * w), int(tip.y * h)
        cv2.circle(display, (fx, fy), 8, (255, 0, 0), -1)

    # B. BEAM
    current_fixed, current_free = None, None
    current_thick = 10
    
    mask = cv2.inRange(hsv, LOWER_BEAM, UPPER_BEAM)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500 and area > max_area:
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), angle = rect
            aspect = max(rw,rh)/min(rw,rh)
            if aspect > 2.0:
                max_area = area
                best_rect = rect
                
    if best_rect:
        box = cv2.boxPoints(best_rect)
        box = np.float32(box)
        dists = []
        for i in range(4):
            for j in range(i+1, 4):
                dists.append((np.linalg.norm(box[i]-box[j]), i, j))
        dists.sort(key=lambda x: x[0], reverse=True)
        p0, p1 = box[dists[0][1]], box[dists[0][2]]
        
        if p0[0] > p1[0]: current_fixed, current_free = p0, p1
        else: current_fixed, current_free = p1, p0
        (cx, cy), (d1, d2), _ = best_rect
        current_thick = min(d1, d2)
        if not locked: cv2.drawContours(display, [np.int0(box)], 0, (100,100,100), 1)

    # C. PHYSICS ENGINE
    active_fixed = ref_fixed if locked else current_fixed
    active_free = ref_free if locked else current_free
    active_thick = ref_thickness if locked else current_thick
    
    sim_deflection = 0      
    sim_load_percent = 0.0  
    load_applied = False    
    status_text = "IDLE"
    
    if active_fixed is not None and active_free is not None:
        p_f = active_fixed.astype(np.float32)
        p_e = active_free.astype(np.float32)
        beam_vec = p_e - p_f
        beam_len = np.linalg.norm(beam_vec)
        if beam_len < 1: beam_unit = np.array([-1.0, 0]) 
        else: beam_unit = beam_vec / beam_len
        
        # Perpendicular Vector (Positive Y = Down)
        perp_unit = np.array([-beam_unit[1], beam_unit[0]])
        if perp_unit[1] < 0: perp_unit = -perp_unit

        if fx is not None:
            finger_vec = np.array([fx, fy]) - p_f
            s = np.dot(finger_vec, beam_unit)       # Distance along beam
            orth_dist = np.dot(finger_vec, perp_unit) # Distance perpendicular
            
            # --- NEW DIRECTION FILTER ---
            # orth_dist < 0 means finger is ABOVE the beam.
            # We ignore if it is significantly above (e.g., -20px).
            
            if 0 <= s <= beam_len + 50:
                if abs(orth_dist) < NEAR_DIST:
                    
                    # Only register interaction if finger is NOT hovering way above
                    if orth_dist > HOVER_THRESHOLD: 
                        status_text = "DETECTED"
                        
                        # TOUCH ZONE
                        if locked and abs(orth_dist) < TOUCH_DIST:
                            # Double check direction: Don't apply load if hovering
                            # We allow a tiny bit of negative (-15) for contact noise
                            if orth_dist > HOVER_THRESHOLD:
                                status_text = "LOAD APPLIED"
                                load_applied = True
                                
                                sim_load_percent = max(0.0, min(1.0, s / beam_len))
                                visual_scale = SIM_BEAM_LEN / beam_len
                                
                                # Only bend if we are pushing DOWN (orth_dist > 0)
                                # If we are in the "noise zone" (-15 to 0), deflection is 0
                                effective_push = max(0, orth_dist) 
                                
                                raw_deflection = effective_push * visual_scale * BEND_SENSITIVITY
                                sim_deflection = min(raw_deflection, MAX_VISUAL_BEND)
                                
                                contact = p_f + s * beam_unit
                                cv2.line(display, (fx,fy), tuple(contact.astype(int)), (0,0,255), 2)

        # D. RENDER AR
        if load_applied and sim_load_percent > 0.1:
            ar_tip_deflection = sim_deflection / visual_scale
            ar_tip_deflection = ar_tip_deflection / (sim_load_percent**2)
            ar_tip_deflection = min(ar_tip_deflection, 200)
        else:
            ar_tip_deflection = 0

        ar_curve = get_cantilever_curve(p_f, p_e, deflection=ar_tip_deflection)
        offset = perp_unit * (active_thick/2)
        poly = np.vstack(((ar_curve+offset).astype(np.int32), np.flipud((ar_curve-offset).astype(np.int32))))
        col = (0,0,255) if load_applied else ((0,255,255) if status_text=="DETECTED" else (0,255,0))
        cv2.fillPoly(display, [poly], col)
        if locked: cv2.circle(display, tuple(p_f.astype(int)), 5, (0,0,255), -1)

    # E. RENDER DIGITAL TWIN
    sim_panel = np.ones((h, SIM_W, 3), dtype=np.uint8) * 255
    
    cv2.rectangle(sim_panel, (WALL_X, BEAM_Y-50), (WALL_X+20, BEAM_Y+50), (50, 50, 50), -1)
    sim_start = np.array([WALL_X, BEAM_Y])                
    sim_end = np.array([WALL_X - SIM_BEAM_LEN, BEAM_Y])   
    
    if not load_applied: sim_tip_deflection = 0 
    else: sim_tip_deflection = sim_deflection / (sim_load_percent**2 + 0.01)
    sim_tip_deflection = min(sim_tip_deflection, MAX_VISUAL_BEND)

    sim_curve_pts = get_cantilever_curve(sim_start, sim_end, deflection=sim_tip_deflection)
    sim_thick_vec = np.array([0, 10]) 
    sim_top = sim_curve_pts - sim_thick_vec
    sim_bot = sim_curve_pts + sim_thick_vec
    sim_poly = np.vstack((sim_top, np.flipud(sim_bot))).astype(np.int32)
    sim_col = (100, 200, 100) 
    if load_applied: sim_col = (100, 100, 255) 
    cv2.fillPoly(sim_panel, [sim_poly], sim_col)
    cv2.polylines(sim_panel, [sim_poly], True, (50,100,50), 2, cv2.LINE_AA)

    if load_applied:
        arrow_idx = int(sim_load_percent * 20)
        arrow_idx = max(0, min(arrow_idx, 20))
        arrow_pos = sim_top[arrow_idx]
        arrow_len = 50 + (sim_deflection * 0.3) 
        arrow_start = (int(arrow_pos[0]), int(arrow_pos[1] - arrow_len))
        arrow_end = (int(arrow_pos[0]), int(arrow_pos[1]))
        
        cv2.arrowedLine(sim_panel, arrow_start, arrow_end, (0, 0, 255), 4, tipLength=0.3)
        cv2.putText(sim_panel, "F", (arrow_start[0]+10, arrow_start[1]+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.line(sim_panel, (WALL_X, BEAM_Y+60), (arrow_end[0], BEAM_Y+60), (0,0,0), 1)
        cv2.line(sim_panel, (arrow_end[0], BEAM_Y+55), (arrow_end[0], BEAM_Y+65), (0,0,0), 1) 
        cv2.putText(sim_panel, f"x={int(sim_load_percent*100)}%", 
                   (arrow_end[0], BEAM_Y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    cv2.putText(sim_panel, "DIGITAL TWIN", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    combined = np.hstack((display, sim_panel))
    if not locked: cv2.putText(combined, "ALIGN & PRESS 'c'", (w-200, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else: cv2.putText(combined, "LOCKED", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("AR Physics", combined)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): 
        if current_fixed is not None:
            ref_fixed, ref_free, ref_thickness, locked = current_fixed, current_free, current_thick, True
    elif key == ord('u'): locked = False

cap.release()
hands.close()
cv2.destroyAllWindows()