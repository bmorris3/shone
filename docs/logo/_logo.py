"""
Generate the shone logo!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from PIL import Image

# Save the logo here:
logo_dir = os.path.dirname(__file__)
uncropped_svg_path = os.path.join(logo_dir, 'logo_uncropped.svg')
cropped_svg_path = os.path.join(logo_dir, 'logo.svg')
png_path = os.path.join(logo_dir, 'logo.png')
ico_path = os.path.join(logo_dir, 'logo.ico')

fig, ax = plt.subplots(dpi=300, figsize=(2, 2))

max_radius = 0.25  # outer circle radius, on the range (0, 1)
min_radius = 0.6  # unit: fraction of `max_radius`

circle_center = (max_radius, max_radius)

color = 'DodgerBlue'

n_annuli = 3
for i in range(n_annuli):
    circle = Circle(
        circle_center,
        radius=max_radius * (min_radius + i / n_annuli * (1 - min_radius)),
        alpha=(i + 1) / 10,
        color=color,
        lw=0
    )
    ax.add_patch(circle)

angle = np.pi / 4  # [rad]
center = (
    circle_center[0] + min_radius * max_radius * np.cos(angle),
    circle_center[1] + min_radius * max_radius * np.sin(angle)
)

angle_opening_width = 4  # [deg]
wedge = Wedge(
    center, r=3,
    theta1=np.degrees(angle) - angle_opening_width / 2,
    theta2=np.degrees(angle) + angle_opening_width / 2,
    color=color,
    lw=0,
    alpha=0.9
)
ax.add_patch(wedge)

ax.axis('off')
ax.set(
    xlim=[0, 1],
    ylim=[0, 1],
    aspect='equal'
)

savefig_kwargs = dict(
    pad_inches=0, transparent=True
)

fig.savefig(uncropped_svg_path, **savefig_kwargs)

# PNG will be at *high* resolution:
fig.savefig(png_path, dpi=800, **savefig_kwargs)

# This is the default matplotlib SVG configuration which can't be easily tweaked:
default_svg_dims = 'width="144pt" height="144pt" viewBox="0 0 144 144"'

# This is a hand-tuned revision to the SVG file that crops the bounds nicely:
custom_svg_dims = 'width="140px" height="140px" viewBox="15 15 120 120"'

# Read the uncropped file, replace the bad configuration with the custom one:
with open(uncropped_svg_path, 'r') as svg:
    cropped_svg_source = svg.read().replace(
        default_svg_dims, custom_svg_dims
    )

# Write out the cropped SVG file:
with open(cropped_svg_path, 'w') as cropped_svg:
    cropped_svg.write(cropped_svg_source)

# Delete the uncropped SVG:
os.remove(uncropped_svg_path)

# Convert the PNG into an ICO file:
img = Image.open(png_path)
img.save(ico_path)
