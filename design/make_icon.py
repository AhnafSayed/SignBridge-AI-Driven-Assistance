"""Generates the SignBridge app icon and splash logo.

Mark: an open hand inside a speech bubble -- the app's job stated
visually (a sign becomes a spoken/written message), rather than relying
on a specific handshape (e.g. the ASL/BdSL "I love you" sign used in an
earlier draft) that not every viewer recognizes as "sign language" at a
glance.

The hand is a proper five-finger fan (index/middle/ring/pinky at
graduated lengths and angles around a centered, longest middle finger,
plus an angled thumb), not the four-equal-straight-fingers mitten from
an earlier draft -- that read as "a hand" but not a *human* one. Getting
the fan right needs fingers rotated around individual pivots spaced far
enough apart that they don't cross (an earlier attempt at fanning got
this wrong and produced a crossed, blob-like mess); each finger is drawn
on its own full-canvas layer and rotated into place around its own
pivot before compositing, which keeps the geometry predictable.

Safe zone: content is kept within roughly the center 50% of the canvas
on each axis (~25% margin all around), inside both the adaptive-icon
safe zone and Android 12's (stricter) splash-icon safe zone.
"""

from PIL import Image, ImageDraw

W = 1024
BLUE = (46, 107, 179, 255)  # 2E6BB3 -- brand primary, background
AMBER = (242, 169, 59, 255)  # F2A93B -- brand accent, the speech bubble
WHITE = (255, 255, 255, 255)  # the hand, on top of the bubble

BUBBLE_BOX = [282, 292, 742, 652]  # x0, y0, x1, y1 -- rounded rect
BUBBLE_RADIUS = 90
TAIL = [(355, 630), (415, 630), (325, 725)]  # triangle, bottom-left tail

PALM_BOX = [442, 480, 582, 580]

# (pivot, angle, length, width). Positive angle = counter-clockwise
# (swings left); negative = clockwise (swings right) -- verified
# empirically against PIL's Image.rotate() convention.
FINGERS = [
    ((466, 482), 10, 150, 26),  # index -- slight inward lean
    ((497, 476), 0, 168, 28),  # middle -- straight, longest
    ((528, 480), -10, 150, 26),  # ring -- slight outward lean
    ((558, 490), -24, 118, 22),  # pinky -- most outward lean, shortest
]
THUMB = ((430, 542), 58, 106, 32)


def capsule_layer(bottom_center, length, width, color):
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = bottom_center
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - length, cy
    d.rounded_rectangle([x0, y0, x1, y1], radius=width / 2, fill=color)
    return layer


def rotated_onto(canvas, layer, angle, pivot):
    rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=False, center=pivot)
    canvas.alpha_composite(rotated)


def build_mark():
    canvas = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)

    # Speech bubble.
    d.rounded_rectangle(BUBBLE_BOX, radius=BUBBLE_RADIUS, fill=AMBER)
    d.polygon(TAIL, fill=AMBER)

    # Hand, in white so it pops off the amber bubble. Fingers and thumb
    # first, palm last so it cleanly covers their base ends.
    for pivot, angle, length, width in FINGERS:
        rotated_onto(canvas, capsule_layer(pivot, length, width, WHITE), angle, pivot)

    thumb_pivot, thumb_angle, thumb_length, thumb_width = THUMB
    rotated_onto(canvas, capsule_layer(thumb_pivot, thumb_length, thumb_width, WHITE), thumb_angle, thumb_pivot)

    palm_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(palm_layer).rounded_rectangle(PALM_BOX, radius=34, fill=WHITE)
    canvas.alpha_composite(palm_layer)

    return canvas


def main():
    fg = build_mark()
    print("content bbox (of 1024):", fg.getbbox())
    fg.save("icon_foreground.png")

    full = Image.new("RGBA", (W, W), BLUE)
    full.alpha_composite(fg)
    full.save("icon_full.png")

    splash = fg.resize((512, 512), Image.LANCZOS)
    splash.save("splash_logo.png")

    print("wrote icon_foreground.png, icon_full.png, splash_logo.png")


if __name__ == "__main__":
    main()
