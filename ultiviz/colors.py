"""Stable BGR class colors and class-name normalization."""

import colorsys


class ColorMixin:
    """Internal implementation mixed into the public Visualizer API."""

    @staticmethod
    def normalize_name(name):
        return " ".join(str(name).lower().replace("-", " ").replace("_", " ").split())

    def color(self, class_id, name=""):
        """Stable class colors, with optional {class_id: BGR} overrides."""
        rgb = colorsys.hsv_to_rgb((int(class_id) * 0.61803398875) % 1, 0.75, 1)
        fallback = tuple(round(channel * 255) for channel in reversed(rgb))
        return tuple(map(int, self.class_colors.get(
            int(class_id), self.PREFERRED_COLORS.get(self.normalize_name(name), fallback)
        )))

    PREFERRED_COLORS = {
        "person": (255, 0, 0), "vest": (0, 165, 255),
        "safety vest": (0, 165, 255), "hat": (180, 105, 255),
        "hardhat": (180, 105, 255), "hard hat": (180, 105, 255),
        "helmet": (180, 105, 255), "safety helmet": (180, 105, 255),
    }
