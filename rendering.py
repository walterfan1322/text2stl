"""Offscreen STL -> multi-view PNG rendering.

Used by the VLM judge loop: render the generated STL from multiple angles,
then pass those images to a vision LLM for semantic verification.

We try pyrender first (cross-platform, needs OpenGL); if unavailable (common
on headless Windows), fall back to matplotlib's 3D projection which works
everywhere but looks cruder — still good enough for "is this recognisable".
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

log = logging.getLogger("text2stl.rendering")

# Four canonical views, chosen to give a VLM enough info to recognise the shape
VIEW_ANGLES = {
    "iso":   (30, 45, 1.8),   # (elevation_deg, azimuth_deg, distance_factor)
    "front": (0, 0, 2.0),
    "side":  (0, 90, 2.0),
    "top":   (89, 0, 2.0),
}

# S5.4: 8-view turntable (every 45°) for richer VLM context.
TURNTABLE_8 = {
    "iso":   (30, 45, 1.8),
    "front": (0, 0, 2.0),
    "fr_r":  (15, 45, 1.9),    # front-right ¾
    "side":  (0, 90, 2.0),
    "ba_r":  (15, 135, 1.9),   # back-right ¾
    "back":  (0, 180, 2.0),
    "ba_l":  (15, 225, 1.9),   # back-left ¾
    "top":   (89, 0, 2.0),
}


def render_stl_views(
    stl_path: Path,
    out_dir: Path,
    resolution: tuple[int, int] = (512, 512),
    use_pyvista: bool = False,
    n_views: int = 4,
) -> list[Path]:
    """Render canonical views of an STL file as PNGs.

    Args:
        use_pyvista: try PyVista first (better lighting / silhouette).
            Falls back to existing trimesh/matplotlib path.
        n_views: 4 (default canonical) or 8 (turntable).

    Returns the list of generated PNG paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    angles = TURNTABLE_8 if n_views >= 8 else VIEW_ANGLES

    if use_pyvista:
        try:
            paths = _render_with_pyvista(stl_path, out_dir, resolution, angles)
            if paths:
                return paths
        except Exception as e:
            log.warning(f"pyvista rendering failed, falling back: {e}")

    try:
        paths = _render_with_trimesh(stl_path, out_dir, resolution, angles)
        if paths:
            return paths
    except Exception as e:
        log.warning(f"trimesh/pyglet rendering failed, falling back to matplotlib: {e}")

    return _render_with_matplotlib(stl_path, out_dir, resolution, angles)


def _render_with_pyvista(
    stl_path: Path, out_dir: Path, resolution: tuple[int, int],
    angles: dict,
) -> list[Path]:
    """High-quality offscreen rendering via PyVista (S5.4)."""
    import numpy as np
    import pyvista as pv

    pv.OFF_SCREEN = True
    mesh = pv.read(str(stl_path))
    if mesh.n_points == 0:
        raise RuntimeError("STL loaded as empty mesh")

    bounds = mesh.bounds
    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])
    extents = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    radius = float(np.linalg.norm(extents)) * 0.5 + 1e-6

    paths: list[Path] = []
    for name, (elev, azim, dist_factor) in angles.items():
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.add_mesh(
            mesh,
            color=(0.78, 0.82, 0.95),
            specular=0.4,
            specular_power=15,
            ambient=0.3,
            diffuse=0.7,
            smooth_shading=True,
            show_edges=True,
            edge_color=(0.25, 0.25, 0.30),
            line_width=0.4,
        )
        plotter.set_background("white")
        plotter.enable_lightkit()

        elev_r = np.deg2rad(elev)
        azim_r = np.deg2rad(azim)
        distance = radius * dist_factor
        cam_pos = center + np.array([
            distance * np.cos(elev_r) * np.sin(azim_r),
            -distance * np.cos(elev_r) * np.cos(azim_r),
            distance * np.sin(elev_r),
        ])
        plotter.camera_position = [cam_pos.tolist(),
                                   center.tolist(),
                                   (0, 0, 1)]

        png = out_dir / f"{name}.png"
        plotter.screenshot(str(png), return_img=False)
        plotter.close()
        paths.append(png)

    return paths


def _render_with_trimesh(
    stl_path: Path, out_dir: Path, resolution: tuple[int, int],
    angles: dict | None = None,
) -> list[Path]:
    """Primary renderer: trimesh.Scene.save_image (uses pyglet/pyopengl)."""
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(stl_path))
    if mesh.is_empty:
        raise RuntimeError("STL loaded as empty mesh")

    extents = mesh.extents
    radius = float(np.linalg.norm(extents)) * 0.5 + 1e-6
    center = mesh.centroid
    angles = angles or VIEW_ANGLES

    paths: list[Path] = []
    for name, (elev, azim, dist_factor) in angles.items():
        scene = trimesh.Scene(mesh.copy())
        distance = radius * dist_factor
        # Build camera transform: place camera at angle looking at center
        elev_r = np.deg2rad(elev)
        azim_r = np.deg2rad(azim)
        cam_pos = center + np.array([
            distance * np.cos(elev_r) * np.sin(azim_r),
            -distance * np.cos(elev_r) * np.cos(azim_r),
            distance * np.sin(elev_r),
        ])
        # trimesh helper: look-at
        transform = trimesh.scene.cameras.look_at(
            points=np.array([center]),
            fov=np.array([60.0, 60.0]),
            rotation=None,
            distance=distance,
            center=center,
        )
        # Override translation with our computed camera position
        transform[:3, 3] = cam_pos
        scene.camera_transform = transform
        scene.camera.resolution = resolution

        png = out_dir / f"{name}.png"
        png_bytes = scene.save_image(resolution=resolution, visible=True)
        if not png_bytes:
            raise RuntimeError("save_image returned empty bytes")
        png.write_bytes(png_bytes)
        paths.append(png)

    return paths


def _render_with_matplotlib(
    stl_path: Path, out_dir: Path, resolution: tuple[int, int],
    angles: dict | None = None,
) -> list[Path]:
    """Fallback renderer: matplotlib 3D. Works without OpenGL."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(stl_path))
    verts = mesh.vertices
    faces = mesh.faces
    angles = angles or VIEW_ANGLES

    paths: list[Path] = []
    for name, (elev, azim, _) in angles.items():
        dpi = 100
        fig = plt.figure(figsize=(resolution[0] / dpi, resolution[1] / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        # Build face polygons
        poly = verts[faces]
        collection = Poly3DCollection(poly, alpha=0.85, edgecolor="gray", linewidth=0.1)
        collection.set_facecolor((0.7, 0.75, 0.9))
        ax.add_collection3d(collection)
        # Limits: bounds
        bmin, bmax = mesh.bounds
        ax.set_xlim(bmin[0], bmax[0])
        ax.set_ylim(bmin[1], bmax[1])
        ax.set_zlim(bmin[2], bmax[2])
        ax.set_box_aspect((bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

        png = out_dir / f"{name}.png"
        fig.savefig(str(png), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        paths.append(png)

    return paths


def encode_png_as_base64(path: Path) -> str:
    """Read a PNG and return its base64-encoded contents (no data: prefix)."""
    return base64.b64encode(Path(path).read_bytes()).decode()


def encode_pngs_as_data_urls(paths: list[Path]) -> list[str]:
    """Convert PNG file paths to data: URLs for OpenAI-compatible image messages."""
    return [
        f"data:image/png;base64,{encode_png_as_base64(p)}"
        for p in paths
    ]
