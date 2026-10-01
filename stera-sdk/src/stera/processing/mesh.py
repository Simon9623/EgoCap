"""Small mesh helpers used by the Rerun visualizer."""

from __future__ import annotations

import numpy as np


def clean_mesh_by_edge_length(
    verts: np.ndarray,
    faces: np.ndarray,
    max_edge_len: float = 0.15,
) -> np.ndarray:
    """Drop triangles with any edge longer than ``max_edge_len`` or zero area.

    Cheap geometric filter used as a baseline cleanup when pymeshlab isn't
    available or wanted. Returns a filtered ``faces`` array; ``verts`` is not
    touched.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    keep = (e0 < max_edge_len) & (e1 < max_edge_len) & (e2 < max_edge_len)
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    keep &= areas > 1e-8
    return faces[keep]


def brighten_colors(
    colors: np.ndarray,
    factor: float = 1.4,
    min_brightness: int = 60,
) -> np.ndarray:
    """Scale brightness then lift darks to ``min_brightness``. Returns uint8."""
    c = colors.astype(np.float32) * factor
    c = np.maximum(c, min_brightness)
    return np.clip(c, 0, 255).astype(np.uint8)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals as the area-weighted average of face normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], face_normals)
    np.add.at(vn, faces[:, 1], face_normals)
    np.add.at(vn, faces[:, 2], face_normals)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (vn / norms).astype(np.float32)
