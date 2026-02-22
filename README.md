# FreenectTD + Skeleton Tracking Extension

This project is a fork of FreenectTD by stosumarte [FreenectTD](https://github.com/stosumarte/FreenectTD)  
by adding experimental **skeleton tracking support**.

> ⚠️ **Experimental Feature – Skeleton Tracking**
> Joint positions are estimated using MediaPipe and may not be fully accurate.
> Additional develop may be required for use.

---

## Skeleton Tracking

- Pose estimation powered by [MediaPipe BlazePose](https://developers.google.com/mediapipe)  
- 33 body landmarks detected  
- RGB-based tracking only (no depth fusion)  
- Joint data exposed to TouchDesigner via CHOP channels  

---

## How to Use

### 1. Enable Skeleton Tracking

Inside the **Custom Freenect TOP**:
Enable Skeleton Tracking → ON


### 2. CHOP 

**CHOP info **:

You will see channels like:
p1/wrist_left:tx
p1/wrist_left:ty
p1/wrist_left:tz

Each joint outputs:

- `tx` → X position  
- `ty` → Y position  
- `tz` → Z position  

---

## Installation

### [RECOMMENDED] Installing using installer

Download the latest installer build from the [releases tab](https://github.com/Jin1025/FreenectTD-skeleton/releases/latest).

Right click on `FreenectTOP_Installer.pkg` and select **"Open"**.

You should now find FreenectTOP under the **"Custom"** OPs panel.

> [!TIP]
> If the Installer gets blocked from running, go to **System Settings > Privacy & Security** and click on **Run Anyway**.

---

## Notes

- Joint coordinates are normalized (0–1 range)  
- For 3D visualization, remap to -1~1 or scene space  
- Depth-based correction is not implemented yet  
- Some jitter may occur due to RGB-only tracking

---
## Contact
email: allaboutmy@snu.ac.kr

---

## Licensing
FreenectTD is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1).
This means you are free to use, modify, and distribute this plugin, including in closed-source applications, provided that any modifications to the plugin itself are released under the same LGPL v2.1 license.

This project also includes the following third-party libraries, each under their respective licenses:

* **libfreenect** – Apache 2.0 License

* **libfreenect2** – Apache 2.0 License

* **libusb** – LGPL 2.1 License

By downloading, using, modifying, or distributing this plugin, either as source-code or binary format, you agree to comply with the terms of both the LGPL-2.1 license for the plugin itself and the respective licenses of the third-party libraries included.
