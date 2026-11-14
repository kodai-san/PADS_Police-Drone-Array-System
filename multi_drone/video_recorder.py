"""
Video recorder — top-down MP4 output using matplotlib + ffmpeg.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os


class VideoRecorder:
    """Records top-down simulation view to MP4."""

    def __init__(self, filepath: str, env_data: dict,
                 fps: int = 30, figsize=(12.8, 7.2), dpi=100):
        self.filepath = filepath
        self.env_data = env_data
        self.fps = fps
        self.figsize = figsize
        self.dpi = dpi
        self.frames = []
        self._fig = None
        self._ax = None
        self._setup_figure()

    def _setup_figure(self):
        self._fig, self._ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        ax = self._ax
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')

        # Draw obstacles
        patches = []
        for obs in self.env_data["obstacles"]:
            poly = MplPolygon(obs, closed=True)
            patches.append(poly)
        if patches:
            pc = PatchCollection(patches, facecolor='#b0b5c0', edgecolor='#707580',
                                 linewidth=0.8, alpha=0.85)
            ax.add_collection(pc)

        # Draw corridor waypoints
        wps = self.env_data["waypoints"]
        ax.plot(wps[:, 0], wps[:, 1], 'k--', alpha=0.3, linewidth=0.8)

        bounds = self.env_data["bounds"]
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)

        # Placeholders for dynamic elements
        self._drone_scatters = None
        self._target_scatter = None
        self._trails = {}
        self._text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                             fontsize=8, verticalalignment='top',
                             fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def add_frame(self, t: float, drone_positions: np.ndarray,
                  target_pos: np.ndarray, node_positions=None,
                  r_des: float = None,
                  extra_text: str = ""):
        """
        Record one frame.
        drone_positions: [N, 2]
        target_pos: [2]
        node_positions: [N, 2] or None
        r_des: desired following distance — draws a dashed circle around target
        """
        ax = self._ax

        # Remove previous dynamic artists
        if self._drone_scatters is not None:
            self._drone_scatters.remove()
        if self._target_scatter is not None:
            self._target_scatter.remove()
        if hasattr(self, '_r_des_circle') and self._r_des_circle is not None:
            self._r_des_circle.remove()
            self._r_des_circle = None

        # Drones
        n = drone_positions.shape[0]
        colors = ['#2ecc71', '#3498db', '#e67e22'][:n]
        self._drone_scatters = ax.scatter(
            drone_positions[:, 0], drone_positions[:, 1],
            c=colors, s=80, zorder=5, edgecolors='k', linewidths=0.5)

        # Target
        self._target_scatter = ax.scatter(
            [target_pos[0]], [target_pos[1]],
            c='red', s=100, marker='*', zorder=6, edgecolors='darkred', linewidths=0.5)

        # R_des following envelope circle
        if r_des is not None:
            circle = plt.Circle((target_pos[0], target_pos[1]), r_des,
                                fill=False, linestyle='--', linewidth=1.0,
                                edgecolor='#e74c3c', alpha=0.5, zorder=3)
            ax.add_patch(circle)
            self._r_des_circle = circle
        else:
            self._r_des_circle = None

        # Formation nodes (if available)
        # drawn as hollow circles
        if hasattr(self, '_node_scatter') and self._node_scatter is not None:
            self._node_scatter.remove()
            self._node_scatter = None
        if node_positions is not None:
            self._node_scatter = ax.scatter(
                node_positions[:, 0], node_positions[:, 1],
                c='none', s=60, edgecolors=colors, linewidths=1.5,
                linestyle='--', zorder=4)
        else:
            self._node_scatter = None

        # Trails (every 10 frames)
        if int(t / 0.05) % 10 == 0:
            for i in range(n):
                key = f'drone_{i}'
                if key not in self._trails:
                    self._trails[key] = []
                self._trails[key].append(drone_positions[i].copy())
                # Draw trail
                trail = np.array(self._trails[key])
                if len(trail) > 1:
                    trail_key = f'trail_line_{i}'
                    if trail_key in self._trails:
                        self._trails[trail_key].remove()
                    line, = ax.plot(trail[:, 0], trail[:, 1], '-',
                                    color=colors[i], alpha=0.3, linewidth=1)
                    self._trails[trail_key] = line

        # Text overlay
        self._text.set_text(f't={t:.1f}s {extra_text}')

        # Render to array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf).copy()
        self.frames.append(img)

    def save(self):
        """Write all frames to MP4 using matplotlib animation or imageio."""
        if not self.frames:
            return

        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)

        try:
            import imageio
            writer = imageio.get_writer(self.filepath, fps=self.fps,
                                         codec='libx264', quality=8)
            for frame in self.frames:
                writer.append_data(frame[:, :, :3])  # RGB only
            writer.close()
        except ImportError:
            # Fallback: save using matplotlib animation
            from matplotlib.animation import FuncAnimation, FFMpegWriter
            fig2, ax2 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax2.axis('off')
            im = ax2.imshow(self.frames[0][:, :, :3])

            def update(i):
                im.set_data(self.frames[i][:, :, :3])
                return [im]

            anim = FuncAnimation(fig2, update, frames=len(self.frames), blit=True)
            writer = FFMpegWriter(fps=self.fps)
            anim.save(self.filepath, writer=writer)
            plt.close(fig2)

        plt.close(self._fig)
        print(f"  Video saved: {self.filepath} ({len(self.frames)} frames)")

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
