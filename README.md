# Pile Driving Hammer Tracker
OAK-D PoE + ChArUco board — drop hammer vertical tracking

---

## What You Need

- OAK-D PoE camera
- PoE switch or PoE injector (~$20 if you don't have one)
- Ethernet cable (laptop → PoE switch → camera)
- Laptop running Linux or Windows
- Printed + laminated ChArUco board (see below)

---

## Step 1 — Print the Board

1. Take `charuco_50cm_300dpi.png` to a print shop
2. Tell them: **"Print at 300 DPI, no scaling, on A1 paper"**
3. Ask for **matte laminate** (not glossy — glare confuses detection)
4. After printing, **measure one black square with a ruler**
   - Expected: ~83mm
   - Open `hammer_tracker.py` and update this line if different:
     ```python
     SQUARE_LENGTH = 0.083   # ← change to your measured value in meters
     MARKER_LENGTH = 0.058   # ← always 70% of SQUARE_LENGTH
     ```

## Step 2 — Mount the Board

- Tape or magnet the board to the **side face of the drop hammer**
- Position it where the camera can see it clearly during the full drop
- Higher on the hammer = less mud/splash exposure

---

## Step 3 — Install Software

```bash
pip install depthai opencv-contrib-python
```

**Linux only** — run once to fix USB/network permissions:
```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Step 4 — Connect the Camera

```
Laptop ──ethernet──► PoE Switch/Injector ──ethernet──► OAK-D PoE
                           ▲
                      wall outlet
```

Camera will get an IP automatically if you have DHCP, or fall back to `169.254.x.x`.

If auto-discovery doesn't work, find the camera IP and set it in the script:
```python
DEVICE_IP = "169.254.1.222"   # your camera's IP
```

Leave as `None` to auto-discover.

---

## Step 5 — Position the Camera

- Mount on a **tripod**, stable, not touching the rig
- Point **perpendicular to the drop direction** (looking at the side of the hammer)
- Keep it **level** — tilt introduces error in height measurement
- Distance: **5–10m** from the hammer works well
- Height: **mid-point of the drop range** gives best tracking

---

## Step 6 — Run

```bash
python hammer_tracker.py
```

### On-screen display
- **Height** — hammer position in meters relative to camera
- **Velocity** — positive = rising, negative = falling
- **Drop avail** — grey = not enough height for next blow to count, green = ready
- **Blows** — running blow count
- Yellow border flash = blow detected

### Keys
- `Q` — quit and save
- `S` — save snapshot image

### Output
- `hammer_log.csv` — saved automatically, one row per frame

---

## CSV Columns

| Column | Meaning |
|---|---|
| timestamp | Seconds since script started |
| height_m | Hammer height in meters (positive = up) |
| velocity_ms_pos_up | m/s — positive rising, negative falling |
| blow | 1 = impact detected this frame, 0 = no |
| set_mm | Penetration in mm on this blow (blank on non-blow frames) |
| inlier_ratio | Detection quality 0–1, below 0.6 = unreliable frame |

The most important column for piling QA is **set_mm** — penetration per blow. As the pile approaches bearing capacity, this number gets smaller.

---

## Tuning on Site

Open `hammer_tracker.py` and adjust these values at the top:

```python
EXPOSURE_US   = 800    # reduce if hammer looks blurry, increase if too dark
MIN_DROP_HEIGHT = 0.30  # lower to 0.15 if missing blows, raise if double-counting
LOCKOUT_SECONDS = 2.0   # seconds to ignore after each blow
```

**If getting missed blows:** lower `MIN_DROP_HEIGHT` to `0.15`
**If getting double counts:** raise `LOCKOUT_SECONDS` to `3.0`
**If board not detected:** reduce `EXPOSURE_US` to `500` and add more lighting

---

## How Blow Detection Works

A blow is counted only when **all three** conditions are true at the same time:

1. **Speed** — hammer moving down faster than 0.5 m/s
2. **Drop height** — hammer has risen at least 30cm since the last blow
   (this is what prevents post-impact bouncing from triggering false blows)
3. **Time lockout** — at least 2 seconds since last blow

---

## Files

| File | Purpose |
|---|---|
| `charuco_50cm_300dpi.png` | Print this — the tracking board |
| `hammer_tracker.py` | Run this — the tracking script |
| `hammer_log.csv` | Generated on run — your data output |