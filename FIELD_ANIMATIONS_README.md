# Field Animation Feature

This document describes how to create animations with velocity and pressure field backgrounds.

## Overview

When running MOMAPPO training with `save_fields=True`, the simulation saves:
- **X-velocity (vx)** field at each timestep
- **Y-velocity (vy)** field at each timestep  
- **Pressure (p)** field at each timestep

These fields are stored in compressed `.npz` format (using float16 for ~50% size reduction).

## How It Works

### 1. Training with Field Saving

The field saving is enabled in `main.py`:

```python
def make_env():
    return SwarmEnv(sim=sim, swarm=swarm, fluid=fluid, inflow=inflow, 
                    folder=folder_name, save_fields=True)
```

During training, fields are saved to: `../runs/{folder_name}/fields/fields_{pid}.npz`

### 2. Field Storage Format

The `.npz` file contains:
- `vx`: X-velocity field array (timesteps, x_resolution, y_resolution) - float16
- `vy`: Y-velocity field array (timesteps, x_resolution, y_resolution) - float16
- `p`: Pressure field array (timesteps, x_resolution, y_resolution) - float16
- `timesteps`: Array of simulation times [seconds]
- `length_x`, `length_y`: Simulation domain dimensions
- `resolution`: Grid resolution tuple

**Example file size for 200 timesteps (1000x40 grid):**
- Without compression: ~194 MB
- With float16 + compression: ~50 MB

### 3. Creating Animations

#### Option A: Automatic (All Three Fields)

Use the helper script to automatically create all three animations:

```bash
python create_field_animations.py --run_folder "../runs/2025-1-1_12-30-45"
```

This creates three MP4 files:
- `animation_vx.mp4` - X-velocity background
- `animation_vy.mp4` - Y-velocity background
- `animation_p.mp4` - Pressure background

#### Option B: Manual (Single Field)

Create animation for a specific field:

```bash
python animate_locations.py \
    --csv "../runs/folder/MOMAPPO/timestamp/locations.csv" \
    --output "animation_vx.mp4" \
    --fields "../runs/folder/fields/fields_12345.npz" \
    --field_type vx \
    --length_x 100.0 \
    --length_y 4.0
```

#### Option C: All Fields Manually

If `--fields` is provided but `--field_type` is not specified, all three animations are created automatically:

```bash
python animate_locations.py \
    --csv "locations.csv" \
    --output "animation.mp4" \
    --fields "fields.npz" \
    --length_x 100.0 \
    --length_y 4.0
```

Creates:
- `animation_vx.mp4`
- `animation_vy.mp4`
- `animation_p.mp4`

## Animation Features

### Visual Properties

1. **Figure Dimensions**: Match simulation aspect ratio (100mm × 4mm = 25:1)
   - Figure size: 25 inches × 4 inches
   - Equal aspect ratio for millimeter units

2. **Field Background**:
   - Viridis colormap for all fields
   - Constant colorbar showing global min/max across all timesteps
   - Smooth interpolation (bilinear)

3. **Swarm Members**:
   - Colored circles (unique color per member)
   - Black center-of-mass indicator
   - Timestamp overlay

4. **Colorbar Labels**:
   - VX: "Velocity X [mm/s]"
   - VY: "Velocity Y [mm/s]"
   - P: "Pressure [mg/(mm·s²)]"

### Performance

- Frame rate: 20 fps (matches 0.05s timestep)
- Output: H264 codec, 200 DPI
- Typical animation time: ~30 seconds for 200 frames

## File Organization

After training and creating animations, your directory structure will be:

```
runs/
└── 2025-1-1_12-30-45/
    ├── fields/
    │   └── fields_12345.npz       # Compressed field data (~50 MB)
    ├── MOMAPPO/
    │   └── 2025-1-1_13-45-30/
    │       ├── locations.csv       # Swarm trajectories
    │       ├── animation_vx.mp4    # X-velocity animation
    │       ├── animation_vy.mp4    # Y-velocity animation
    │       └── animation_p.mp4     # Pressure animation
    └── ...
```

## Advanced Usage

### Custom Parameters

```bash
python animate_locations.py \
    --csv "locations.csv" \
    --output "custom.mp4" \
    --fields "fields.npz" \
    --field_type vx \
    --length_x 100.0 \
    --length_y 4.0 \
    --radius 0.25 \
    --fps 20
```

### Without Fields

To create basic animations without field backgrounds:

```bash
python animate_locations.py \
    --csv "locations.csv" \
    --output "basic.mp4" \
    --length_x 100.0 \
    --length_y 4.0
```

## Memory Considerations

For 40,000 timesteps with 1000×40 resolution:
- Raw float32: ~38 GB
- Compressed float16: ~10 GB

**Recommendation**: For very long simulations, consider:
1. Reducing save frequency (save every Nth step)
2. Using lower resolution for visualization
3. Processing in chunks

## Troubleshooting

### "No fields file found"
- Ensure `save_fields=True` was set during training
- Check that training completed successfully
- Look for `.npz` file in `runs/{folder}/fields/`

### "Memory Error"
- Reduce number of timesteps
- Use lower resolution
- Process in smaller chunks

### "Animation looks pixelated"
- Increase DPI in `animate_locations.py` (line with `dpi=200`)
- Use higher resolution field data

### "Wrong dimensions"
- Verify `--length_x` and `--length_y` match simulation
- Check saved values in `.npz` file:
  ```python
  import numpy as np
  data = np.load('fields.npz')
  print(data['length_x'], data['length_y'])
  ```

## Technical Details

### Coordinate System
- **X-axis**: Flow direction (0 to length_x mm)
- **Y-axis**: Channel height (0 to length_y mm)
- **Field arrays**: Stored as [timestep, x, y]
- **Visualization**: Transposed to [y, x] for proper display

### Colormap Choice
Viridis was chosen because:
- Perceptually uniform
- Colorblind-friendly
- Good contrast for scientific data
- Shows gradients clearly

## Example Workflow

```bash
# 1. Run training with field saving
python main.py
# ... select 'y' for new folder
# ... training runs with save_fields=True

# 2. After training completes, create animations
python create_field_animations.py --run_folder "../runs/2025-1-9_12-30-45"

# 3. View animations
# Open animation_vx.mp4, animation_vy.mp4, animation_p.mp4
```

## Performance Tips

1. **FFmpeg**: Ensure FFmpeg path is correct in `animate_locations.py` line 13
2. **GPU Acceleration**: Training uses GPU, animation rendering uses CPU
3. **Parallel Creation**: Create animations in parallel:
   ```bash
   python animate_locations.py ... --field_type vx &
   python animate_locations.py ... --field_type vy &
   python animate_locations.py ... --field_type p &
   ```

## Citation

If you use these visualizations in publications, consider mentioning:
- Viridis colormap (matplotlib)
- FFmpeg for video encoding
- PhiFlow for fluid simulation

