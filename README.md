# FreenectTD + Skeleton Tracking Extension

This project extends [FreenectTD](https://github.com/stosumarte/FreenectTD)  
by adding experimental **skeleton tracking support**.

⚠️ **Note:**  
Joint positions are currently not fully accurate and may require additional filtering or calibration.

---

## Skeleton Tracking

- Pose estimation powered by [MediaPipe BlazePose](https://developers.google.com/mediapipe)  
- 33 body landmarks detected  
- RGB-based tracking only (no depth fusion)  
- Joint data exposed to TouchDesigner via CHOP channels  

---

## Detected Landmarks (33)

1. nose  
2. left_eye_inner  
3. left_eye  
4. left_eye_outer  
5. right_eye_inner  
6. right_eye  
7. right_eye_outer  
8. left_ear  
9. right_ear  
10. mouth_left  
11. mouth_right  
12. left_shoulder  
13. right_shoulder  
14. left_elbow  
15. right_elbow  
16. left_wrist  
17. right_wrist  
18. left_pinky  
19. right_pinky  
20. left_index  
21. right_index  
22. left_thumb  
23. right_thumb  
24. left_hip  
25. right_hip  
26. left_knee  
27. right_knee  
28. left_ankle  
29. right_ankle  
30. left_heel  
31. right_heel  
32. left_foot_index  
33. right_foot_index  

---

## How to Use

### 1. Enable Skeleton Tracking

Inside the **Custom Freenect TOP/COMP**:
Enable Skeleton Tracking → ON

---
Operator → customFreenect

You will see channels like:
p1/left_wrist:tx
p1/left_wrist:ty
p1/left_wrist:tz
p1/left_wrist:tracked

Each joint outputs:

- `tx` → X position  
- `ty` → Y position  
- `tz` → Z position  
- `tracked` → confidence value  

---

## Notes

- Joint coordinates are normalized (0–1 range)  
- For 3D visualization, remap to -1~1 or scene space  
- Depth-based correction is not implemented yet  
- Some jitter may occur due to RGB-only tracking  

---

## Limitations

- No depth alignment  
- Single-person tracking optimized (multi-person may not work correctly)  
- Joint position accuracy depends on lighting and camera angle  
