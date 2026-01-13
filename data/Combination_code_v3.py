# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 20:05:52 2024

@author: Taieb Belfekih
"""

import cv2
import os
import copy
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Change default save format to .jpg
mpl.rcParams['savefig.format'] = 'jpg'

import networkx as nx
import pickle
from skimage.morphology import thin, skeletonize
import sknw
import pandas as pd
from skimage.transform import radon, rotate
from scipy.signal import argrelmax, peak_widths, peak_prominences
import math
from scipy.optimize import curve_fit

import pickle
from anastruct import SystemElements

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

plt.rcParams.update({'font.size': 20})
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# Positive value moves the board lower limit UP (removes some of the lower plate)
BOARD_LOWER_OFFSET_PX = 20 
LOWER_LINER_MARGIN_PX = 2

from sklearn.cluster import KMeans
from itertools import zip_longest

from scipy.ndimage import gaussian_filter1d

# Global rotation angle (in degrees). None means "not computed yet".
ROTATION_ANGLE = 0.19622093023255616
MAX_TRACK_DIST = 10  # or 80, tune as you like

from scipy.ndimage import gaussian_filter1d  # already imported at top


from collections import defaultdict
import math
import networkx as nx

def _polygon_area_xy(points_xy):
    """Shoelace formula, |area| of polygon given as [[x,y], ...]."""
    if len(points_xy) < 3:
        return 0.0
    pts = np.asarray(points_xy, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _analyze_cycles(G):
    """Return list of {'id','nodes','len','area'} for all simple cycles."""
    und = G.to_undirected()
    basis = nx.cycle_basis(und)
    infos = []
    for cid, cyc in enumerate(basis):
        if len(cyc) < 3:
            continue
        pts_xy = [(G.nodes[n]["o"][1], G.nodes[n]["o"][0]) for n in cyc]
        # polygon area (shoelace)
        pts = np.asarray(pts_xy, dtype=float)
        x = pts[:, 0]; y = pts[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        infos.append({
            "id": cid,
            "nodes": cyc,
            "len": len(cyc),
            "area": float(area),
        })
    return infos

def remove_pure_triangles(G, max_iter=20):
    """
    Remove triangles whose edges are NOT used by any other cycle.
    For each such triangle, remove one edge (shortest), then
    contract its endpoints if they become degree-2 and are not
    part of any 4+ node cycle.

    This respects:
      - do NOT remove edges that support bigger cycles
        (no open big cycles),
      - triangles that share all edges with bigger cycles
        are kept (they are 'structural').
    """
    if G.number_of_nodes() == 0:
        return

    for it in range(max_iter):
        infos = _analyze_cycles(G)
        if not infos:
            print(f"[triangles] iter {it}: no cycles, stop.")
            break

        # triangles and 4+ cycles
        triangles = [c for c in infos if c["len"] == 3]
        big_cycles = [c for c in infos if c["len"] >= 4]

        if not triangles:
            print(f"[triangles] iter {it}: no triangles, stop.")
            break

        # build edge -> cycles map
        edge2cycles = defaultdict(set)
        for c in infos:
            cyc_nodes = c["nodes"]
            m = len(cyc_nodes)
            for k in range(m):
                u = cyc_nodes[k]
                v = cyc_nodes[(k+1) % m]
                key = tuple(sorted((u, v)))
                edge2cycles[key].add(c["id"])

        # edges belonging to 4+ cycles (we must not remove these)
        big_ids = {c["id"] for c in big_cycles}

        # find candidate edges: edges that belong only to a triangle,
        # not to any 4+ cycle
        candidates = []
        tri_by_id = {c["id"]: c for c in triangles}
        for tri in triangles:
            cyc_nodes = tri["nodes"]
            m = len(cyc_nodes)
            for k in range(m):
                u = cyc_nodes[k]
                v = cyc_nodes[(k+1) % m]
                key = tuple(sorted((u, v)))
                cycles_here = edge2cycles[key]
                # Only this triangle and nothing else
                if cycles_here == {tri["id"]}:
                    candidates.append((tri["id"], u, v))

        if not candidates:
            print(f"[triangles] iter {it}: no 'pure' triangle edges; stop.")
            break

        # pick the shortest candidate edge to remove
        def edge_len(tri_id_u_v):
            _, u, v = tri_id_u_v
            y1, x1 = G.nodes[u]['o']
            y2, x2 = G.nodes[v]['o']
            return math.hypot(x2 - x1, y2 - y1)

        tri_id, u, v = min(candidates, key=edge_len)
        print(f"[triangles] iter {it}: removing edge {u}-{v} "
              f"from triangle {tri_id}")

        if G.has_edge(u, v):
            G.remove_edge(u, v)

        # build node -> big-cycle membership (before contraction)
        node2big = defaultdict(set)
        for c in big_cycles:
            cid = c["id"]
            for n in c["nodes"]:
                node2big[n].add(cid)

        # helper: contract a degree-2 node if not in any big cycle
        def contract_node(n):
            if n not in G:
                return
            if G.degree(n) != 2:
                return
            if node2big[n]:
                return  # don't touch nodes belonging to big cycles
            nbrs = list(G.neighbors(n))
            if len(nbrs) != 2:
                return
            n1, n2 = nbrs
            if n1 == n2:
                G.remove_node(n)
                return
            if not G.has_edge(n1, n2):
                y1, x1 = G.nodes[n1]['o']
                y2, x2 = G.nodes[n2]['o']
                pts = np.array([[y1, x1],
                                [y2, x2]], dtype=np.int32)
                G.add_edge(n1, n2, pts=pts)
            G.remove_node(n)

        # after removing the edge, endpoints may become degree-2;
        # contract them if they are not part of any big cycle
        contract_node(u)
        contract_node(v)

    print("[triangles] done:", G.number_of_nodes(), "nodes,",
          G.number_of_edges(), "edges")
    
def remove_all_degree_one_nodes(G):
    """
    Remove ALL degree-1 nodes (leaves), iteratively.
    This does NOT touch degree-2 nodes or cycles.
    """
    if G.number_of_nodes() == 0:
        return

    changed = True
    while changed:
        changed = False
        for n, deg in list(G.degree()):
            if deg == 1:
                G.remove_node(n)
                changed = True


def simplify_small_cycles_edge_based(
        G,
        small_area_ratio=0.25,
        big_area_ratio=0.7,
        max_iter=30,
        show_hist_initial=True,
        show_hist_each_iter=False,
        big_growth_tol=1.5):
    """
    Iteratively remove edges belonging to very small cycles (including triangles).

    Logic:
      - Compute a 'typical trapezoid area' A_med from cycles with len >= 4.
      - 'Small' cycles: area < small_area_ratio * A_med.
      - 'Big' cycles: len >= 4 AND area >= big_area_ratio * A_med.
      - At each iteration:
          * Pick the smallest small cycle.
          * Choose an edge (u,v) in that cycle such that:
              - that edge is NOT used by any big cycle, and
              - among such edges it is the shortest.
          * Remove that edge.
          * Contract its endpoints if they become degree-2 nodes that do not
            belong to any big cycle.
      - Stop if:
          * no small cycles remain, OR
          * no safe edge exists for the smallest small cycle, OR
          * big cycles grow in area above big_growth_tol * initial_big_max.

    This guarantees:
      - big (trapezoid) cycles never open,
      - big cycle areas never blow up unexpectedly,
      - triangles and tiny 4-cycles are removed first.
    """

    if G.number_of_nodes() == 0:
        print("[small-cycles] empty graph")
        return

    # --- initial cycle stats to define A_med, A_small, A_big ---
    infos0 = _analyze_cycles(G)
    big_support = [c for c in infos0 if c["len"] >= 4]
    if not big_support:
        print("[small-cycles] no 4+ cycles; nothing to simplify safely.")
        return

    areas4 = np.array([c["area"] for c in big_support])
    A_med = float(np.median(areas4))
    A_small = small_area_ratio * A_med
    A_big = big_area_ratio * A_med

    if show_hist_initial:
        plt.figure(figsize=(6, 4))
        plt.hist([c["area"] for c in infos0], bins=20)
        plt.xlabel("Cycle area [px²]")
        plt.ylabel("Count")
        plt.title("Histogram of cycle areas (initial)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    big_initial = [c["area"] for c in big_support if c["area"] >= A_big]
    A_big_baseline = max(big_initial) if big_initial else A_med

    print(f"[small-cycles] A_med={A_med:.1f}, "
          f"A_small<{A_small:.1f}, A_big>={A_big:.1f}")

    # --- main iteration loop ---
    for it in range(max_iter):
        infos = _analyze_cycles(G)
        if not infos:
            print(f"[small-cycles] iter {it}: no cycles, stop.")
            break

        # cycles of any length >=3 used to find small ones (incl. triangles)
        cycles_all = [c for c in infos if c["len"] >= 3]
        # only 4+ used as "big / trapezoid" candidates
        cycles_4p = [c for c in infos if c["len"] >= 4]

        if not cycles_all:
            print(f"[small-cycles] iter {it}: no valid cycles, stop.")
            break

        if show_hist_each_iter:
            plt.figure(figsize=(6, 4))
            plt.hist([c["area"] for c in cycles_all], bins=20)
            plt.xlabel("Cycle area [px²]")
            plt.ylabel("Count")
            plt.title(f"Cycle area histogram (iter {it})")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        small_cycles = [c for c in cycles_all if c["area"] < A_small]
        big_cycles = [c for c in cycles_4p if c["area"] >= A_big]

        if not small_cycles:
            print(f"[small-cycles] iter {it}: no small cycles left, stop.")
            break

        big_ids = {c["id"] for c in big_cycles}
        big_areas = [c["area"] for c in big_cycles] or [0.0]
        if max(big_areas) > big_growth_tol * A_big_baseline:
            print(f"[small-cycles] iter {it}: big cycle area grew too much "
                  f"({max(big_areas):.1f} > {big_growth_tol}*{A_big_baseline:.1f}), stop.")
            break

        # map edges -> cycles they belong to (all cycles >=3)
        edge2cycles = defaultdict(set)
        for c in cycles_all:
            cyc_nodes = c["nodes"]
            m = len(cyc_nodes)
            for k in range(m):
                u = cyc_nodes[k]
                v = cyc_nodes[(k + 1) % m]
                key = tuple(sorted((u, v)))
                edge2cycles[key].add(c["id"])

        # smallest small cycle
        c_small = min(small_cycles, key=lambda c: c["area"])
        cyc_nodes = c_small["nodes"]
        m = len(cyc_nodes)

        # candidate edges: edges on this small cycle that are NOT used by any big cycle
        candidate_edges = []
        for k in range(m):
            u = cyc_nodes[k]
            v = cyc_nodes[(k + 1) % m]
            key = tuple(sorted((u, v)))
            if edge2cycles[key].isdisjoint(big_ids):
                candidate_edges.append((u, v))

        if not candidate_edges:
            print(f"[small-cycles] iter {it}: smallest small cycle has no "
                  "edge free of big cycles; stop.")
            break

        # choose shortest candidate edge (prefer top short edge in triangles, etc.)
        def edge_len(uv):
            u, v = uv
            y1, x1 = G.nodes[u]['o']
            y2, x2 = G.nodes[v]['o']
            return math.hypot(x2 - x1, y2 - y1)

        e_best = min(candidate_edges, key=edge_len)
        u, v = e_best
        print(f"[small-cycles] iter {it}: removing edge {u}-{v} "
              f"(len={edge_len(e_best):.1f}, small area={c_small['area']:.1f})")

        if G.has_edge(u, v):
            G.remove_edge(u, v)

        # map node -> big cycle membership BEFORE contraction
        node2big = defaultdict(set)
        for c in big_cycles:
            cid = c["id"]
            for n in c["nodes"]:
                node2big[n].add(cid)

        def contract_node(n):
            if n not in G:
                return
            if G.degree(n) != 2:
                return
            if node2big[n]:
                return  # don't touch structural/big-cycle nodes
            nbrs = list(G.neighbors(n))
            if len(nbrs) != 2:
                return
            n1, n2 = nbrs
            if n1 == n2:
                G.remove_node(n)
                return
            if not G.has_edge(n1, n2):
                y1, x1 = G.nodes[n1]['o']
                y2, x2 = G.nodes[n2]['o']
                pts = np.array([[y1, x1],
                                [y2, x2]], dtype=np.int32)
                G.add_edge(n1, n2, pts=pts)
            G.remove_node(n)

        contract_node(u)
        contract_node(v)

    print("[small-cycles] done:",
          G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")


def compress_degree2_paths(G, border_x_tol=15):
    """
    Build a new graph where:
      - 'key' nodes are preserved:
          * nodes whose original degree != 2, or
          * nodes near the left/right borders (within border_x_tol).
      - any chain of degree-2 nodes between two key nodes is collapsed
        to a single edge with a pts array along the path.

    Returns a NEW graph H.
    """

    if G.number_of_nodes() == 0:
        return G

    # horizontal borders via x (column)
    xs = [data["o"][1] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)

    def is_border(n):
        x = G.nodes[n]["o"][1]
        return (x <= min_x + border_x_tol) or (x >= max_x - border_x_tol)

    # key nodes = structural or border nodes
    key_nodes = set()
    for n in G.nodes():
        deg = G.degree(n)
        if deg != 2 or is_border(n):
            key_nodes.add(n)

    # new graph with only key nodes
    H = nx.Graph()
    for n in key_nodes:
        H.add_node(n, o=G.nodes[n]["o"])

    visited_edges = set()

    for u in key_nodes:
        for v in G.neighbors(u):
            edge0 = tuple(sorted((u, v)))
            if edge0 in visited_edges:
                continue

            path_nodes = [u]
            curr = v
            prev = u
            visited_edges.add(edge0)

            # follow chain of degree-2 nodes
            while curr not in key_nodes:
                path_nodes.append(curr)
                nbrs = list(G.neighbors(curr))
                if len(nbrs) != 2:
                    break  # something weird; stop
                nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
                edge = tuple(sorted((curr, nxt)))
                if edge in visited_edges:
                    break
                visited_edges.add(edge)
                prev, curr = curr, nxt

            if curr not in path_nodes:
                path_nodes.append(curr)

            # both ends must be key nodes
            u2 = path_nodes[0]
            w = path_nodes[-1]
            if u2 not in key_nodes or w not in key_nodes:
                continue
            if u2 == w:
                continue

            # build pts along path
            pts = []
            for n in path_nodes:
                y, x = G.nodes[n]["o"]
                pts.append([y, x])
            pts = np.array(pts, dtype=np.int32)

            if not H.has_edge(u2, w):
                H.add_edge(u2, w, pts=pts)

    return H

def prune_degrees(G, border_x_tol=15):
    """
    Enforce:
      - no isolated nodes (deg 0),
      - no degree-1 nodes away from borders,
      - no degree-2 nodes away from borders
        (those are contracted into a single edge).

    Modifies G in place.
    """

    if G.number_of_nodes() == 0:
        return

    xs = [data["o"][1] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)

    def is_border(n):
        x = G.nodes[n]["o"][1]
        return (x <= min_x + border_x_tol) or (x >= max_x - border_x_tol)

    # 1) remove degree-0 nodes
    for n, deg in list(G.degree()):
        if deg == 0:
            G.remove_node(n)

    # 2) iteratively remove degree-1 nodes away from borders
    changed = True
    while changed:
        changed = False
        for n, deg in list(G.degree()):
            if deg == 1 and not is_border(n):
                G.remove_node(n)
                changed = True

    # 3) contract interior degree-2 nodes
    again = True
    while again:
        again = False
        for n, deg in list(G.degree()):
            if deg != 2:
                continue
            if is_border(n):
                continue  # allow degree-2 at borders
            nbrs = list(G.neighbors(n))
            if len(nbrs) != 2:
                continue
            u, v = nbrs
            if u == v:
                G.remove_node(n)
                again = True
                continue
            # add edge with pts between u and v
            if not G.has_edge(u, v):
                y1, x1 = G.nodes[u]["o"]
                y2, x2 = G.nodes[v]["o"]
                pts = np.array([[y1, x1],
                                [y2, x2]], dtype=np.int32)
                G.add_edge(u, v, pts=pts)
            G.remove_node(n)
            again = True


def simplify_graph_by_cycles(
        G,
        border_x_tol=15,
        large_area_ratio=0.5,
        min_cycle_len_for_trap=4,
        angle_corner_deg=20,
        max_iter=2,
        show_hist=True):
    """
    Cycle-based graph simplification for the corrugated-board structure.

    Idea:
      - Find cycles and compute their polygon areas.
      - 'Large' non-triangular cycles ≈ trapezoids.
      - Keep nodes that:
          * are corners of at least one large cycle, OR
          * have degree >= 3, OR
          * are degree-2 border nodes (left/right ends).
      - Remove other degree-1/2 nodes; for degree-2 nodes, reconnect neighbours.

    Parameters
    ----------
    border_x_tol : int
        Margin in pixels near the left/right image borders that we consider
        "border region" where we keep degree-2 nodes.
    large_area_ratio : float
        A cycle is considered 'large' if its area >= large_area_ratio * median
        area of all non-triangular cycles.
    min_cycle_len_for_trap : int
        Minimum number of nodes in a cycle to be considered non-triangular.
    angle_corner_deg : float
        Deviation from 180° (in degrees) used to decide if a node is a
        'corner' inside a large cycle. Larger -> fewer corners.
    max_iter : int
        How many simplification passes to run.
    show_hist : bool
        If True, plot a histogram of all cycle areas in the first iteration.
    """

    if G.number_of_nodes() == 0:
        print("[cycles] empty graph, nothing to do.")
        return

    xs = [data["o"][1] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)

    def is_border_node(n):
        x = G.nodes[n]["o"][1]
        return (x <= min_x + border_x_tol) or (x >= max_x - border_x_tol)

    angle_corner_rad = np.deg2rad(angle_corner_deg)

    for it in range(max_iter):
        und = G.to_undirected()
        cycles = nx.cycle_basis(und)

        if not cycles:
            print(f"[cycles] iter {it}: no cycles found, stop.")
            break

        cycle_infos = []
        for cyc in cycles:
            if len(cyc) < 3:
                continue
            pts_xy = [(G.nodes[n]["o"][1], G.nodes[n]["o"][0]) for n in cyc]
            area = _polygon_area_xy(pts_xy)
            cycle_infos.append({"nodes": cyc, "area": area, "len": len(cyc)})

        if not cycle_infos:
            print(f"[cycles] iter {it}: no valid cycles, stop.")
            break

        # --- histogram of all cycle areas (for inspection) ---
        if show_hist and it == 0:
            areas_all = np.array([c["area"] for c in cycle_infos])
            plt.figure(figsize=(6, 4))
            plt.hist(areas_all, bins=20)
            plt.xlabel("Cycle area [px²]")
            plt.ylabel("Count")
            plt.title("Histogram of all cycle areas (frame 0)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        non_tri = [c for c in cycle_infos if c["len"] >= min_cycle_len_for_trap]
        if not non_tri:
            print(f"[cycles] iter {it}: no non-triangular cycles, stop.")
            break

        areas_non_tri = np.array([c["area"] for c in non_tri])
        median_area = np.median(areas_non_tri)
        area_thresh = large_area_ratio * median_area

        large_cycles = [c for c in non_tri if c["area"] >= area_thresh]

        print(f"[cycles] iter {it}: total={len(cycle_infos)}, "
              f"non-tri={len(non_tri)}, large={len(large_cycles)}, "
              f"median_area={median_area:.1f}, thresh={area_thresh:.1f}")

        # --- 1. mark 'corner' nodes in large cycles ---
        corner_nodes = set()
        for c in large_cycles:
            cyc = c["nodes"]
            m = len(cyc)
            for k in range(m):
                n_prev = cyc[k - 1]
                n_cur = cyc[k]
                n_next = cyc[(k + 1) % m]

                p_prev = np.array([G.nodes[n_prev]["o"][1],
                                   G.nodes[n_prev]["o"][0]], dtype=float)
                p_cur = np.array([G.nodes[n_cur]["o"][1],
                                  G.nodes[n_cur]["o"][0]], dtype=float)
                p_next = np.array([G.nodes[n_next]["o"][1],
                                   G.nodes[n_next]["o"][0]], dtype=float)

                v1 = p_prev - p_cur
                v2 = p_next - p_cur
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    corner_nodes.add(n_cur)
                    continue

                cosang = np.dot(v1, v2) / (n1 * n2)
                cosang = np.clip(cosang, -1.0, 1.0)
                ang = np.arccos(cosang)   # angle between edges at node

                # straight ~180°, so we treat as 'corner' when far from 180°
                if abs(np.pi - ang) > angle_corner_rad:
                    corner_nodes.add(n_cur)

        # --- 2. nodes we must keep ---
        keep_nodes = set(corner_nodes)
        for n, deg in G.degree():
            if deg >= 3:
                keep_nodes.add(n)           # structural junctions
            elif deg == 2 and is_border_node(n):
                keep_nodes.add(n)           # specimen borders

        # --- 3. removable candidates: deg<=2 not in keep_nodes ---
        candidates = [n for n in list(G.nodes())
                      if n not in keep_nodes and G.degree(n) <= 2]

        if not candidates:
            print(f"[cycles] iter {it}: no removable nodes, stop.")
            break

        print(f"[cycles] iter {it}: removing {len(candidates)} nodes")

        # --- 4. remove candidates, reconnect degree-2 nodes ---
        for n in candidates:
            if n not in G:
                continue
            nbrs = list(G.neighbors(n))
            if len(nbrs) == 2:
                u, v = nbrs
                if u != v and not G.has_edge(u, v):
                    y1, x1 = G.nodes[u]["o"]
                    y2, x2 = G.nodes[v]["o"]
                    pts = np.array([[y1, x1],
                                    [y2, x2]], dtype=np.int32)
                    G.add_edge(u, v, pts=pts)
            # degree 0 or 1: just remove
            G.remove_node(n)

    print("[cycles] simplification done:",
          G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
    
def simplify_graph_trapezoids(G, border_x_tol=15, max_iter=10):
    """
    Simple graph 'denoising' tailored to the corrugated-board pattern.

    We want to keep:
      - nodes with degree >= 3 (junctions / trapezoid corners),
      - nodes with degree 2 at the left/right borders of the board
        (these are the end nodes of the structure).

    We treat as 'noise':
      - degree-1 nodes away from borders (small branches),
      - degree-2 nodes away from borders (subdivision points along edges).

    For degree-2 nodes we *contract* them:
      - connect their two neighbours directly,
      - remove the node.

    This preserves the global trapezoid structure but removes
    unnecessary intermediate nodes.
    """

    if G.number_of_nodes() == 0:
        return

    # approximate horizontal borders of the graph using node x-coordinates (columns)
    xs = [data["o"][1] for _, data in G.nodes(data=True)]
    min_x, max_x = min(xs), max(xs)

    for _ in range(max_iter):
        removed_any = False

        # --- 1) remove degree-1 nodes away from borders (small dangling branches) ---
        to_remove_deg1 = []
        for n, deg in list(G.degree()):
            if deg != 1:
                continue
            x = G.nodes[n]["o"][1]

            # keep degree-1 nodes only if they are at the extreme left/right borders
            if x <= min_x + border_x_tol or x >= max_x - border_x_tol:
                continue

            to_remove_deg1.append(n)

        if to_remove_deg1:
            G.remove_nodes_from(to_remove_deg1)
            removed_any = True

        # --- 2) contract degree-2 nodes away from borders ---
        to_remove_deg2 = []
        for n, deg in list(G.degree()):
            if deg != 2:
                continue

            x = G.nodes[n]["o"][1]
            # keep degree-2 nodes only at borders
            if x <= min_x + border_x_tol or x >= max_x - border_x_tol:
                continue

            neighbours = list(G.neighbors(n))
            if len(neighbours) != 2:
                continue

            u, v = neighbours
            if u == v:
                continue

            # reconnect neighbours along the same "edge"
            if not G.has_edge(u, v):
                # Build an artificial pts array using the endpoints
                y1, x1 = G.nodes[u]['o']
                y2, x2 = G.nodes[v]['o']
                pts = np.array([[y1, x1],
                                [y2, x2]], dtype=np.int32)
                G.add_edge(u, v, pts=pts)
            
            to_remove_deg2.append(n)


        if to_remove_deg2:
            G.remove_nodes_from(to_remove_deg2)
            removed_any = True

        if not removed_any:
            break



def correct_vertical_shadow(horizontal_image,
                            image_upper_lim_rough,
                            image_lower_lim_rough,
                            smooth_sigma=3,
                            shadow_factor=0.6,
                            max_gain=3.0,
                            show_plots=False):
    """
    Compensate vertical illumination shadow by brightening darker rows
    between image_upper_lim_rough and image_lower_lim_rough.

    - Compute mean intensity per row in that band.
    - Smooth it.
    - Rows that are much darker than the typical level are scaled up.
    - Scaling is limited by max_gain.

    Parameters
    ----------
    horizontal_image : 2D uint8 array
        Rotated & trimmed grayscale image (frame 0).
    image_upper_lim_rough, image_lower_lim_rough : int
        Rough top/bottom rows of the board in this image.
    smooth_sigma : float
        Sigma for Gaussian smoothing of row means.
    shadow_factor : float
        Rows with mean < shadow_factor * target_level are "shadowed".
    max_gain : float
        Maximum multiplicative gain.
    show_plots : bool
        If True, plot original vs corrected row means.

    Returns
    -------
    corrected : 2D uint8 array
        Shadow-corrected image.
    """

    # Copy as float
    img_f = horizontal_image.astype(np.float32)

    # Clip band to image size
    h = img_f.shape[0]
    r0 = max(0, min(image_upper_lim_rough, h-1))
    r1 = max(r0+1, min(image_lower_lim_rough, h))

    band = img_f[r0:r1, :]
    row_means = band.mean(axis=1)

    # Smooth the profile to avoid reacting to noise
    row_means_smooth = gaussian_filter1d(row_means, sigma=smooth_sigma)

    valid = row_means_smooth > 0
    if not np.any(valid):
        # Nothing to do
        return horizontal_image

    # Typical level = median of nonzero rows
    target_level = np.median(row_means_smooth[valid])
    shadow_thresh = shadow_factor * target_level

    gains = np.ones_like(row_means_smooth, dtype=np.float32)
    shadow_idx = row_means_smooth < shadow_thresh
    gains[shadow_idx] = target_level / np.maximum(row_means_smooth[shadow_idx], 1.0)
    gains = np.clip(gains, 1.0, max_gain)

    # Apply gains row by row in that band
    corrected = img_f.copy()
    for i, g in enumerate(gains):
        r = r0 + i
        corrected[r, :] *= g

    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    if show_plots:
        # Plot original vs corrected row means (same kind of curve as before)
        corr_means = corrected[r0:r1, :].mean(axis=1)
        x = np.arange(r0, r1)

        plt.figure(figsize=(10, 6))
        plt.plot(x, row_means, label='original')
        plt.plot(x, corr_means, label='corrected')
        plt.title('Row-wise intensity before/after shadow correction')
        plt.xlabel('Row number')
        plt.ylabel('Averaged value of pixels')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return corrected

def get_rotation_angle(gray_image,
                       image_upper_lim_rough,
                       image_lower_lim_rough,
                       use_manual=False,
                       manual_angle=-0.44,
                       show_plots=False):
    """
    Compute (once) and cache the rotation angle.

    - If use_manual is True, just use manual_angle.
    - Otherwise, use sinogram_analysis on the first frame.

    Subsequent calls just return the cached ROTATION_ANGLE.
    """
    global ROTATION_ANGLE

    if ROTATION_ANGLE is not None:
        return ROTATION_ANGLE

    if use_manual:
        ROTATION_ANGLE = manual_angle
        print(f"Using manual rotation angle: {ROTATION_ANGLE:.3f} deg")
    else:
        masked_square_image = reshape_and_mask(gray_image,
                                               image_upper_lim_rough,
                                               image_lower_lim_rough,
                                               show_plots=show_plots)
        ROTATION_ANGLE = sinogram_analysis(masked_square_image,
                                           show_plots=show_plots)
        print(f"Rotation angle calculated from sinogram: {ROTATION_ANGLE:.3f} deg")

    return ROTATION_ANGLE


#%% Define output Folder names, input variables, procedures to do methods for image skeletonization

# sample name
sample_name = 'Exp3'
#sample_name = 'Probe1_22'
plt.ioff() 


# path of the Excel file
file_path_zwick = 'Load_Deformation_Curves.xlsx'

# Read the sheet corresponding to a Specimen from the Excel file and save it 
# the measurement data in a data frame
df = pd.read_excel(file_path_zwick, sheet_name='Ex3')


# create a forlder for this sample if not already exsiting
probe_output_folder = f"data_{sample_name}"
if not os.path.exists(probe_output_folder):
    os.makedirs(probe_output_folder)


# number the fps value will be divided by
fps_div = 10

# choose the skimage morphological skeletonization function:    
    # True: skeletonize
    # False : thin
Ske_thin_method = True

if Ske_thin_method:
    meth = "ske"
else:
    meth = "thin"


test_info = f"{sample_name}_{meth}_rate{fps_div}"
test_folder = f"{probe_output_folder}/{test_info}"

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# folder to save the graph figures
destination_folder_graphs = f'{test_folder}/graphs_figs_{test_info}'
if not os.path.exists(destination_folder_graphs):
    os.makedirs(destination_folder_graphs)


# folder to save the graph figures
destination_folder_extra = f'{test_folder}/extra_figs_{test_info}'
if not os.path.exists(destination_folder_extra):
    os.makedirs(destination_folder_extra)


# extract new Frames or work with ready existing frame folder
new_frame_extraction = False


# modeling and clustering: If True then modeling and clustering are conducted
Node_clustering = True
board_modeling = True

# if model_elasticity = False then the board will be modeled in all the frames in the compression phase
model_elasticity = True
model_all = False

if not model_elasticity and not model_all:
    print('Give the frame limits of the model manually. From Frame x to Frame y')
    # elasticity for fps_div = 10  is between frames 39 and 43
    model_ref_frame = None
    model_end_frame = None

    model_ref_frame = int(input('Enter Frame x : '))

    model_end_frame = int(input('Enter Frame y : '))


# do a new image analysis or use already existing displacement excel files and graph .pkl files
# if Flase give the compression_start and end_compression variables manually
# to used then in the modelling and clustering parts
new_image_analysis = True
USE_MANUAL_COMPRESSION_FRAMES = True   # set to True to override
MANUAL_COMPRESSION_START = 20          # your chosen start frame
MANUAL_COMPRESSION_END   = None        # or an int if you want to override end too

image_analysis_output_file_path = f'{test_folder}/img_analysis_{test_info}.xlsx'  # Update the path if needed

'''if not new_image_analysis:
    
    print('Variables generated during image analysis are missing. Synchronization, clustering and modeling can not be done')
    print('Start a new image analysis if needed to sava these varibales in an excel file')
    # Load the Excel file
    image_analysis_output_df = pd.read_excel(image_analysis_output_file_path, sheet_name='img_analy_data')
    
    print(image_analysis_output_df)
    compression_start = int(image_analysis_output_df.loc[0,'compression_start'])
    compression_end = int(image_analysis_output_df.loc[0,'compression_end'])
    frame_pos_peak_D = int(image_analysis_output_df.loc[0,'frame_pos_peak_D'])
    frame_buckling_end = int(image_analysis_output_df.loc[0,'frame_buckling_end'])
    delta_H_pixel_image = int(image_analysis_output_df.loc[0,'delta_H_pixel_image'])
    thickness_0_pixels_img = int(image_analysis_output_df.loc[0,'thickness_0_pixels_img'])
    thickness_min_pixel_img = int(image_analysis_output_df.loc[0,'thickness_min_pixel_img'])



elif new_image_analysis : 
    
    # if true safe the figures of graphs overlayed on the video frames in the output folder
    save_figures_global = True'''

save_figures_global = True
# Define the nodes to be used in the model. is must start by node 1 and max is 42
nodes_limits = [1,10]


# sample dimensions in mm

# width of the sampels in CD
b = 25                        # in mm
sample_length = 100 

sample_area =  sample_length * b



#%% Frames extraction from the FCT sample video

########### modified function with 205 Frames ###########
def extract_frames(video_path, output_folder, extraction_rate):
    """
    Extract frames either from a video file OR from a folder of images.

    Parameters
    ----------
    video_path : str
        Path to the video file OR to a folder containing images.
    output_folder : str
        Path of the folder where to store the frames as frame_0000.jpg, ...
    extraction_rate : int
        Keep 1 every 'extraction_rate' frames/images.

    Sets
    ----
    fps_original : float
        Frame rate of the original source (used later to compute time).
        - If video_path is a folder, fps_original is set manually.
    """
    import glob
    global fps_original

    # Case 1: video_path is a folder containing screenshots
    if os.path.isdir(video_path):
        # >>>>> SET THIS TO THE REAL FPS OF THE ORIGINAL RECORDING IF YOU KNOW IT <<<<<
        # If you don't know the FPS, set it to 1.0 and "time" will mean "frames"
        fps_original = 1.0

        os.makedirs(output_folder, exist_ok=True)

        # Get list of images in the folder (png/jpg/jpeg/tif)
        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(video_path, ext)))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No image files found in folder: {video_path}")

        print(f"Found {len(image_paths)} images in {video_path}")
        print(f"Copying every {extraction_rate}-th image into {output_folder}")

        extracted_frame_count = 0
        for idx, img_path in enumerate(image_paths):
            if idx % extraction_rate != 0:
                continue

            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: could not read {img_path}, skipping.")
                continue

            frame_filename = os.path.join(
                output_folder,
                f"frame_{extracted_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, image)
            extracted_frame_count += 1

        print(f"Done. Extracted {extracted_frame_count} frames from folder {video_path}.")
        return

    # Case 2: video_path is a real video file (original behavior)
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get original fps from the video
    fps_original = video_capture.get(cv2.CAP_PROP_FPS)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Extracting frames from video {video_path} into {output_folder}")
    print(f"Original FPS: {fps_original}")

    frame_count = 0
    extracted_frame_count = 0

    success, image = video_capture.read()
    while success:
        if frame_count % extraction_rate == 0:
            frame_filename = os.path.join(
                output_folder,
                f"frame_{extracted_frame_count:04d}.jpg"
            )
            cv2.imwrite(frame_filename, image)
            extracted_frame_count += 1

        frame_count += 1
        success, image = video_capture.read()

    video_capture.release()
    print(f"Done. Extracted {extracted_frame_count} frames from video {video_path}.")        

# Define the paths for the function input
video_path = f"Sample_videos/{sample_name}" 

# Create a directory to store the image if it doesn't exist
frames_folder = f"{probe_output_folder}/Frames_{sample_name}_rate{fps_div}"

if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

# call function to extract frames from the sample video
extract_frames(video_path, frames_folder, fps_div)

new_fps = fps_original / fps_div



#%% Image analysis: generation of graphs, node tracking

def image_filtering(image, 
                    gaussian_kernel_size = 7, #5
                    medianBlur_parameter = 5, #3
                    threshold = 30, #25
                    min_size = 150,
                    show_plots = False):
    """
    applies a filtering process and threshold a gray image
    
    Parameters
    ----------
    image : Array
        DESCRIPTION.
    gaussian_kernel_size : int
        size of the gaussian filtering/ blurring kernel.
    threshold : int
        binary threshold of the image.
    min_size : int
        size of the minimum area of conencted white pixels.

    Returns
    -------
    smoothed_image : Array
        filtered binary image.

    """

    # # Show the original image
    # plt.figure(figsize= (fig_size, fig_size))
    # plt.imshow(image, "gray", vmin=0, vmax=255)
    # plt.axis('off')
    # plt.title('original image')
    # plt.show()

    ##########################################################################
    # Step 1: Gaussian Blurring
    blurred = cv2.GaussianBlur(image, (gaussian_kernel_size, gaussian_kernel_size), 1)
    
    # # Show the filtered thresh result
    # plt.figure(figsize= (fig_size, fig_size))
    # plt.imshow(blurred, "gray", vmin=0, vmax=255)
    # plt.axis('off')
    # plt.title('blurred image')
    # plt.show()
    
    
    ##########################################################################
    # Step 2: thresholding / binarization
    
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    
    
    #########################################################
    # Trying the opening filter
    
    # # Remove small objects and noise using morphological operations
    # kernel = np.ones((3, 3), np.uint8)
    # morph_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # # Show the filtered thresh result
    # plt.figure(figsize= (fig_size, fig_size))
    # plt.imshow(morph_opened, "gray", vmin=0, vmax=255)
    # plt.axis('off')
    # plt.title(f'binary image with threshold = {threshold}')
    # plt.show()
    
    ##########################################################################
    # Step 3: crate mask and filter by connected components
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                                                binary, 
                                                                connectivity=8)
    # Create an output image to store the filtered components
    filtered_image = np.zeros(binary.shape, dtype=np.uint8)

    # Iterate through all connected components
    for label_i in range(1, num_labels):  # Skipping the background label 0
        area = stats[label_i, cv2.CC_STAT_AREA]

        # Filter by component size
        if area >= min_size:
            filtered_image[labels == label_i] = 255
    
    # # Show the by connected componants filtered thresh result
    # plt.figure(figsize= (fig_size, fig_size))
    # plt.imshow(filtered_image, "gray", vmin=0, vmax=255)
    # plt.axis('off')
    # plt.title('filtering by connected components')
    # plt.show()
    
    ##########################################################################
    # Apply Median Blurring to the binary image. It smoothens the image without
    # causing discontinuities. This replaces an opening that creates discontinuities 
    # in the paper
    
    smoothed_image = cv2.medianBlur(filtered_image, medianBlur_parameter)
    
    if show_plots:
        
        # Show the filtered thresh result
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(smoothed_image, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'binary filtered image with threshold = {threshold}')
        plt.show()
    
    return smoothed_image


def fft_freq_detec(image):
    
    # Apply FFT
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)  # Shift the zero-frequency component to the center

    # Get magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)

    # Plot the original image and its magnitude spectrum
    plt.figure(figsize=(12, 6))

    plt.subplot(211)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(212)
    plt.imshow(np.log1p(magnitude_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum (Log scale)')

    plt.show()

    # Detect dominant frequency
    # Find the coordinates of the peak in the magnitude spectrum
    indices = np.unravel_index(np.argmax(magnitude_spectrum, axis=None), magnitude_spectrum.shape)
    peak_frequency = (indices[0] - magnitude_spectrum.shape[0] // 2, indices[1] - magnitude_spectrum.shape[1] // 2)

    # Calculate the frequency in terms of cycles per image width/height
    frequency_x = peak_frequency[1] / image.shape[1]
    frequency_y = peak_frequency[0] / image.shape[0]

    print(f"Peak frequency detected at: {peak_frequency}")
    print(f"Frequency in x direction: {frequency_x:.2f} cycles per image width")
    print(f"Frequency in y direction: {frequency_y:.2f} cycles per image height")
    
    return(frequency_x)


def rough_workspace_limits(gray_image):
    """
    Define a rough workspace around the board with the board being centered
    in this work space


    Parameters
    ----------
    gray_image : array
        raw gray image input.

    Returns
    -------
    image_upper_lim_rough : TYPE
        DESCRIPTION.
    image_lower_lim_rough : TYPE
        DESCRIPTION.

    """
    
    # Initialize a list to store the sum of pixel values for each row
    row_sums_list = []
    
    # Iterate through the rows from the last row to the first row
    for row in gray_image:
        row_sums_list.append(np.sum(row)/gray_image.shape[1])
    
    # converting list to array
    row_sums_arr = np.array(row_sums_list)
    
    # make a copy and filter it
    # Optional to minimize the number of peaks by filtering the significantly small
    # values where some small local peaks can be detected
    row_sums_arr[row_sums_arr < 0.01 * np.max(row_sums_arr)] = 0
    
    #### search of the peaks (local maximums) extract the 2 highest peak that 
    #### represent an approximation of the liners
    
    # Find the index/x-values of local/relative  maximims/peaks 
    # with parameter "order" set to 10
    peaks_index_row_wise = argrelmax(row_sums_arr, order=10)[0]
    
    # initialise an empty array for the y-values of the peaks found
    peak_values = np.zeros(len(peaks_index_row_wise))
    
    # calculate the y-values of the peaks found
    for i, peak_index in enumerate(peaks_index_row_wise):
        peak_values[i] = row_sums_arr[peak_index]
        
    # Combine arrays column-wise to match the peak index to its y-value
    peak_coord = np.column_stack((peaks_index_row_wise, peak_values)).astype(int)
    
    # Sort the array by y-values (second column, index 1)
    peak_coord = peak_coord[peak_coord[:, 1].argsort()][::-1]
    
    # Extract the coordinates of the 2 peaks having the maximum y-values 
    peak_coord_max_2 = peak_coord[:2, :]
    
    upper_liner_center_pos = min(peak_coord_max_2[:,0])
    lower_liner_center_pos = max(peak_coord_max_2[:,0])
    
    # distance between the approximated center of the liners, which is an 
    # approximation of corrugated board thickness
    dist_liners_center_line = abs(peak_coord_max_2[0,0] - peak_coord_max_2[1,0])
    
    # --- NEW: find the valley between the two strongest peaks ---
    i_min = min(peak_coord_max_2[:, 0])
    i_max = max(peak_coord_max_2[:, 0])
    
    # values between the two peaks (inclusive)
    row_segment = row_sums_arr[i_min:i_max+1]
    
    # index of minimum in that segment (relative to i_min)
    valley_rel_idx = np.argmin(row_segment)
    valley_idx = i_min + valley_rel_idx  # absolute row index of valley
   
    # set the row limits
    image_upper_lim_rough = round( upper_liner_center_pos - dist_liners_center_line)
    image_lower_lim_rough = round( lower_liner_center_pos + dist_liners_center_line)

    # --- CLAMP TO IMAGE BOUNDS AND ENFORCE ORDER ---
    h = gray_image.shape[0]

    # clamp to [0, h-1]
    image_upper_lim_rough = max(0, min(image_upper_lim_rough, h - 2))
    image_lower_lim_rough = max(0, min(image_lower_lim_rough, h - 1))

    # ensure upper < lower; if not, fall back to full height
    if image_lower_lim_rough <= image_upper_lim_rough:
        image_upper_lim_rough = 0
        image_lower_lim_rough = h - 1

    return image_upper_lim_rough, image_lower_lim_rough




def reshape_and_mask(gray_image, 
                     image_upper_lim_rough, 
                     image_lower_lim_rough, 
                     show_plots=False):
    """ 
    centeration of the board in a square array and filter it with a circular mask
    necessary for a better sinogram analysis and results
    
    Parameters
    ----------
    gray_image : TYPE
        DESCRIPTION.
    show_plots : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    masked_square_image : TYPE
        DESCRIPTION.

    """

    # Crop the board from the image
    cropped_board_image = gray_image[ image_upper_lim_rough:image_lower_lim_rough, :]
    
    # Define the dimensions of the larger square array where to center the board
    square_size = max(gray_image.shape)
    
    # Create a square array with zeros
    large_image = np.zeros((square_size, square_size), dtype=np.uint8)
    
    # Get the dimensions of the corpped board image
    small_height, small_width = cropped_board_image.shape
    
    # Calculate the top-left corner of the placement
    # and place the smaller image in the center of the larger array
    large_image[(square_size - small_height) // 2 : 
                    (square_size - small_height) // 2 + small_height, 
                (square_size - small_width) // 2 : 
                    (square_size - small_width) // 2 +small_width] = cropped_board_image
    
    # Create a circular mask with zeros
    mask = np.zeros((square_size, square_size), dtype=np.uint8)
    center = (square_size // 2, square_size // 2)
    radius = min(center[0], center[1])
    cv2.circle(mask, center, radius, (255), thickness=-1)
    
    # Apply the circular mask to the larger image using bitwise AND
    masked_square_image = cv2.bitwise_and(large_image, mask)
    
    
    # # draw the circular mask limit
    
    if show_plots:
        
        masked_square_image_show_mask = masked_square_image.copy()
        cv2.circle(masked_square_image_show_mask, center, radius, (255), thickness=1)
        
        # Show the masked large image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(masked_square_image_show_mask, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title('square image with mask boarder')
        plt.show()
        
        plt.close()
        
        
        masked_square_image_show_mask = masked_square_image.copy()
        # Create the inverse mask
        inverse_mask = cv2.bitwise_not(mask)
        # Set all values outside the circle to 255 in the image
        masked_square_image_show_mask = cv2.bitwise_or(large_image, inverse_mask)

        
        # Show the masked large image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(masked_square_image_show_mask, "gray", vmin=0, vmax=255)
        plt.axis('off')
        # plt.title('Reshaped square image')
        plt.show()
        
        plt.close()
        
        
    return masked_square_image




# Radon transform of the original gray image, to calculate the exact angle of 
# direction of the board and then rotate it to make it horizental

def sinogram_analysis(masked_square_image, show_plots= False):
    """
    (for better precision and results an angle correction may be needed)
    """

    resolution_coeff = 2
    
    # Calculate the angle of the ramp using the Radon transform
    theta = np.linspace(0., 180., round(max(masked_square_image.shape)*resolution_coeff), endpoint=False)
    
    # the total number of steps
    angle_steps_number = len(theta)
    
    sinogram = radon(masked_square_image, theta=theta, circle = True)
    
    # reverse the rows of the sinogram --> same direction as in the image (Top to bottom)
    sinogram =  sinogram[::-1, :]
    
    # Find the index of the angle corresponding to the maximum value in the Radon transform
    # which is the x position of the maximum in the sinogram
    max_t_index, max_angle_index = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    
    # read the corresponding angle in the array theta
    max_angle = theta[max_angle_index]
    
    # # small angle correction if needed, it must be manually added to the 
    ### rotation angle :((
    # Angle_corr_factor = 0.1
    # Angle_correction = - Angle_corr_factor *(90 - max_angle)

    # calculate the rotation angle
    # ( = difference between the direction angle and the horizantal )
    rotation_angle = (90 - max_angle)
    
    ##########################################################################
    ### determin the liner's positions and limits using the y-axis/t-axis of the sinogram
    ### another possibility that will not be used 
    
    # # Define the number of columns to be extracted from the sinogram 
    # # from each side of the column containing the maximum peak
    # col_range = 2

    # # extract the columns array from the sinogram
    # sinogram_cols_ext = sinogram[ : , max_angle_index - col_range : max_angle_index + col_range+1]

    # # calculate the row-wise sum of the extracted array
    # sinogram_cols_sum = np.sum(sinogram_cols_ext,axis=1)

    # # find peaks (local maximums) with a given condition "order" to reduce the
    # # number of small irrelevent peaks
    # peaks_from_sinogram = argrelmax(sinogram_cols_sum, order=10)

    # # Create a figure and axis to plot the columns extracted from the sinogram
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(sinogram_cols_ext)
    # ax.axis('on')
    # ax.set_title("Radon transform (Sinogram)\n columns extracted")
    # ax.set_xlabel("Projection angle (degrees)")
    # ax.set_ylabel("Projection position (pixels)")
    # plt.show()

    # # Visualize the row-wise sum of the extratcted array
    # plt.figure(figsize=(10, 6))
    # plt.plot(sinogram_cols_sum)
    # plt.title('Sum of Pixel Values for Each Row (Bottom to Top)')
    # plt.xlabel('Row Number (from top to bottom)')
    # plt.ylabel('Sum of Pixel Values')
    # plt.show()
    
    
    if show_plots:

        # Visualizing the sinogram and then a zoom around the maximum peak
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        ax.imshow(sinogram)
        ax.axis('on')
        ax.set_title("Radon transform\n(Sinogram)")
        
        ax.set_xlabel("Projection angle in degrees")
        ax.set_ylabel("Projection position in pixels")
        
        # Draw a vertical line showing the maximum x-position, corresponding to the 
        # index of the maximum angle in the original scale and to the angle itself 
        # in the second scale
        ax.axvline(x=max_angle_index, color='red', linewidth=1)
        
        # Draw a horizontal line showing the maximum y-position,
        ax.axhline(y=max_t_index, color='red', linewidth=1)
        
        #### Setting the labels to show the angle values in degrees instead of the 
        #### index values
        
        # Set the number of ticks in a way that the angles are well represented
        # coeff = 1 --->  0, 10, 20, 30, 40     Degrees
        # coeff = 2 --->  0,  5, 10, 15, 20     Degrees
        ticks_coeff = 1
        num_ticks = 18*ticks_coeff+1
        
        # define the tick positions corresponding to the index range
        tick_positions = np.linspace(0, angle_steps_number, num_ticks)
        # define the new labels showing the angle values in degrees and round the values
        tick_labels = np.round(np.linspace(0, 180, num_ticks) , decimals=0).astype(int)
        
        # Apply the customised ticks and labels to show the degree values instead 
        # of index values
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        # Automatically optimize plot size
        # plt.tight_layout()
        
        # show the sinogram
        plt.show()
        
        # show the plot
        plt.close()
        
        
        
        #############################################################################
        ############################# zooming in the sinogram around the maximum peak
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        ax.imshow(sinogram)
        ax.axis('on')
        ax.set_title("Radon transform\n(Sinogram)")
        
        ax.set_xlabel("Projection angle in degrees")
        ax.set_ylabel("Projection position in pixels")
        
        # Draw a vertical line showing the maximum x-position, corresponding to the 
        # index of the maximum angle in the original scale and to the angle value 
        # in degrees in the customized scale (ticks and labels)
        ax.axvline(x=max_angle_index, color='red', linewidth=1)
        
        # Draw a horizontal line showing the maximum y-position,
        ax.axhline(y=max_t_index, color='red', linewidth=1)
        
        #### Setting the labels to show the angle values in degrees instead of the 
        #### index values
        
        # Set the number of ticks in a way that the angles are represented in a uniform way
        # coeff = 1 --->  0, 10, 20, 30     Degrees
        # coeff = 2 --->  0,  5, 10, 15     Degrees
        ticks_coeff = 20
        num_ticks = 18*ticks_coeff+1
        
        # define the tick positions corresponding to the index range
        tick_positions = np.linspace(0, angle_steps_number , num_ticks)
        # define the new labels showing the angle values in degrees and round the values
        tick_labels = np.round(np.linspace(0, 180, num_ticks) , decimals=1)
        
        # Apply the customised ticks and labels to show the degree values instead 
        # of index values
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        # choose the range of degrees around the maximum to be plottet in the x axis
        degree_range = 1
        # set limits to the axes aroud the maximum point
        ax.set_xlim(max_angle_index- round(angle_steps_number/180*degree_range), 
                    max_angle_index+ round(angle_steps_number/180*degree_range)
                    )
        
        ax.set_ylim(max_t_index- 10, 
                    max_t_index+ 10
                    )
        
        # Automatically optimize plot size
        #plt.tight_layout()
        
        # show the sinogram
        plt.show()
        
        # show the plot
        plt.close()

    return rotation_angle




def row_analysis_fct_frame_0(horizontal_image,
                             image_upper_lim_rough,
                             image_lower_lim_rough,
                             show_plots = True):
    
    # Initialize a list to store the sum of pixel values for each row
    row_sums_list = []
    
    # Iterate through the rows of the image between the two rough limits of the board
    for row in horizontal_image[image_upper_lim_rough : image_lower_lim_rough , : ]:
        row_sums_list.append(np.sum(row)/horizontal_image.shape[1])
        
        
    # converting list to array
    row_sums_arr = np.array(row_sums_list)
    
    # concatinate arrays of zeros to keep the same lengh of the image and the 
    # same index and positions
    
    row_sums_arr = np.concatenate((np.zeros(image_upper_lim_rough), 
                                  row_sums_arr,
                                  np.zeros(horizontal_image.shape[0]-image_lower_lim_rough)
                                  ))
    
    # make a copy and filter it
    # Optional to minimize the number of peaks by filtering the significantly small
    # values where some small local peaks can be detected
    row_sums_arr[row_sums_arr < 0.01 * np.max(row_sums_arr)] = 0
    
    #### search of the peaks (local maximums) extract the 2 highest peak that 
    #### represent an approximation of the liners
    
    # Find the index/x-values of local/relative  maximims/peaks 
    # with parameter "order" set to 10
    peaks_index_row_wise = argrelmax(row_sums_arr, order=10)[0]
    
    # initialise an empty array for the y-values of the peaks found
    peak_values = np.zeros(len(peaks_index_row_wise))
    
    # calculate the y-values of the peaks found
    for i, peak_index in enumerate(peaks_index_row_wise):
        peak_values[i] = row_sums_arr[peak_index]
        
    # Combine arrays column-wise to match the peak index to its y-value
    peak_coord = np.column_stack((peaks_index_row_wise, peak_values)).astype(int)
    
    # Sort the array by y-values (second column, index 1)
    peak_coord = peak_coord[peak_coord[:, 1].argsort()][::-1]
    
    # Extract the coordinates of the 2 peaks having the maximum y-values 
    peak_coord_max_2 = peak_coord[:2, :]
    
    upper_liner_center_pos = min(peak_coord_max_2[:,0])
    lower_liner_center_pos = max(peak_coord_max_2[:,0])
    
    
    ##### calculate the prominences of peaks, distance between top and base line
    prominences = peak_prominences(row_sums_arr, peak_coord_max_2[:,0])
    # calculate the heights of the contour/base of the peaks
    contour_heights = peak_coord_max_2[:,1] - prominences[0]
    
    
    ##############################################################################
    # possible improvement of the choce of the parameter "rel_height" based on how
    # large the base of a peak is. +
    # But her also some problems can accure like in this example:
    # The height goes down to zero and then, the base is maximumized. Here the other
    # shorter height must be take. a function that does this??
    
    
    ##### Calculate information about the widths of the peaks in a given 
    # relative height of the peak set to 0.5
    ##### this gives an estimation of the liners thickness
    peaks_width_info = peak_widths(row_sums_arr,  
                             peak_coord_max_2[:,0],
                             rel_height=0.2,
                             prominence_data = prominences
                             )
     
    
    peaks_width_info_100 = peak_widths(row_sums_arr,  
                             peak_coord_max_2[:,0],
                             rel_height=1,
                             prominence_data = prominences
                             )
    
    # The minimum thickness of the two liners is chosen
    liners_thickness_estimation = round(min(peaks_width_info[0]))
    
    Flute_center_line = (peak_coord_max_2[0,0] + peak_coord_max_2[1,0]) //2
    
    # distance between the approximated center of the liners, which is an 
    # approximation of corrugated board thickness
    dist_liners_center_line = abs(peak_coord_max_2[0,0] - peak_coord_max_2[1,0])
    
    # --- NEW: find the valley between the two strongest peaks ---
    i_min = min(peak_coord_max_2[:, 0])
    i_max = max(peak_coord_max_2[:, 0])

    # values between the two peaks (inclusive)
    row_segment = row_sums_arr[i_min:i_max+1]

    # index of minimum in that segment (relative to i_min)
    valley_rel_idx = np.argmin(row_segment)
    valley_idx = i_min + valley_rel_idx  # absolute row index of valley
    
    # calculate an estimation of the upper limit of the board 
    board_upper_limit_estimation = upper_liner_center_pos - liners_thickness_estimation//2
    # calculate an estimation of the lower limit of the board 
    board_lower_limit_estimation = lower_liner_center_pos + liners_thickness_estimation//2
    
    # calculate an estimation of the upper limit of the board 
    board_upper_limit_estimation = upper_liner_center_pos - liners_thickness_estimation//2

    # NEW: set the lower limit of the board around the valley, not at the very bright plate edge
    BOARD_VALLEY_OFFSET = -1  # tweak: -1, 0, or +1 rows depending on what looks best
    board_lower_limit_estimation = valley_idx + BOARD_VALLEY_OFFSET
    
    if show_plots:
       # after the existing plotting code in this function:
       plt.axvline(x=valley_idx, color='purple', linestyle='-.', label='Valley (mask bottom)')
       plt.legend()
       
    # calculate an estimation of the immer limits of the liners 
    upper_liner_inner_limit_estimation = upper_liner_center_pos + liners_thickness_estimation//2
    # calculate an estimation of the lower limit of the board 
    lower_liner_inner_limit_estimation = lower_liner_center_pos - liners_thickness_estimation//2
    
    # estimation of the board thickness
    Board_thickness_estimation = abs(board_upper_limit_estimation - board_lower_limit_estimation)+1
    
    flute_center_line_pos = upper_liner_center_pos + ( lower_liner_center_pos - upper_liner_center_pos)/2
    
    # set the row limits for less iterations in the comming analysis
    image_upper_lim = round( upper_liner_center_pos - 0.5 * dist_liners_center_line)
    image_lower_lim = round( lower_liner_center_pos + 0.5 * dist_liners_center_line)
    
    
    if show_plots:
        # Visualize the output
        plt.figure(figsize=(16, 10))
        plt.plot(row_sums_arr)
        plt.title('Averaged value of pixels in each row')
        plt.xlabel('Row number')
        plt.ylabel('Averaged value of pixels')
        
        # plot an x on the 2 highest peaks
        # plt.plot(peak_coord_max_2[:,0], peak_coord_max_2[:,1], "x")
        
        # # plot vertical lines representing the 2 peaks
        # plt.vlines(x = peak_coord_max_2[:,0], 
        #            ymax = peak_coord_max_2[:,1], 
        #            ymin = np.zeros(len(peak_coord_max_2[:,0])),
        #            color='green', linewidth=1)
        
        # plot the peaks prominences
        plt.vlines(x=peak_coord_max_2[:,0], 
                   ymin=contour_heights, 
                   ymax=peak_coord_max_2[:,1],
                   color="orange" , linewidth=2, label="Peaks' prominence")
        
        
        # plot horizontal lines representing the widths of peaks
        plt.hlines(*peaks_width_info[1:], color="brown" , linewidth=2, label="Liners' thickness")
        
        # plot horizontal lines representing the widths of peaks
        plt.hlines(*peaks_width_info_100[1:], color="green", linestyle='--' , linewidth=2, label="Base width" )
        
        # Draw vertical lines at the left and right width limits

        left_width = peaks_width_info[2][0]
        right_width = peaks_width_info[3][1]
        plt.axvline(x=left_width, color='r', linestyle='--', linewidth=2, label='Board limits')
        plt.axvline(x=right_width, color='r', linestyle='--', linewidth=2 )
        
        plt.axvline(x=flute_center_line_pos, color='b', linestyle='--', linewidth=2, label='Fluting centerline')
        
        # zoom in the x axis
        plt.xlim(110,200)
        plt.legend()
        plt.tight_layout()
        plt.grid("on")
        # show the final plot
        plt.show()
        # close the plot
        plt.close()

    return (upper_liner_center_pos,             lower_liner_center_pos,
            board_upper_limit_estimation ,      board_lower_limit_estimation,
            upper_liner_inner_limit_estimation, lower_liner_inner_limit_estimation,
            image_upper_lim,                    image_lower_lim,
            
            liners_thickness_estimation,
            Flute_center_line,
            dist_liners_center_line,
            Board_thickness_estimation)


def row_analysis_fct_next_frame(horizontal_image,
                                image_upper_lim,
                                image_lower_lim,
                                liners_thickness_estimation,
                                show_plots = False):
    
    # Initialize a list to store the sum of pixel values for each row
    row_sums_list = []
    
    # Iterate through the rows of the image between the two rough limits of the board
    for row in horizontal_image[image_upper_lim : image_lower_lim , : ]:
        row_sums_list.append(np.sum(row)/horizontal_image.shape[1])
        
        
    # converting list to array
    row_sums_arr = np.array(row_sums_list)
    
    # concatinate arrays of zeros to keep the same lengh of the image and the 
    # same index and positions
    
    row_sums_arr = np.concatenate((np.zeros(image_upper_lim),
                                  row_sums_arr,
                                  np.zeros(horizontal_image.shape[0]-image_lower_lim)
                                  ))
    
    # make a copy and filter it
    # Optional to minimize the number of peaks by filtering the significantly small
    # values where some small local peaks can be detected
    row_sums_arr[row_sums_arr < 0.01 * np.max(row_sums_arr)] = 0
    
    #### search of the peaks (local maximums) extract the 2 highest peak that 
    #### represent an approximation of the liners
    
    # Find the index/x-values of local/relative  maximims/peaks 
    # with parameter "order" set to 10
    peaks_index_row_wise = argrelmax(row_sums_arr, order=10)[0]
    
    # initialise an empty array for the y-values of the peaks found
    peak_values = np.zeros(len(peaks_index_row_wise))
    
    # calculate the y-values of the peaks found
    for i, peak_index in enumerate(peaks_index_row_wise):
        peak_values[i] = row_sums_arr[peak_index]
        
    # Combine arrays column-wise to match the peak index to its y-value
    peak_coord = np.column_stack((peaks_index_row_wise, peak_values)).astype(int)
    
    # Sort the array by y-values (second column, index 1)
    peak_coord = peak_coord[peak_coord[:, 1].argsort()][::-1]
    
    # Extract the coordinates of the 2 peaks having the maximum y-values 
    peak_coord_max_2 = peak_coord[:2, :]
    
    upper_liner_center_pos = min(peak_coord_max_2[:,0])
    lower_liner_center_pos = max(peak_coord_max_2[:,0])
    
    
    ##### calculate the prominences of peaks, distance between top and base line
    prominences = peak_prominences(row_sums_arr, peak_coord_max_2[:,0])
    # calculate the heights of the contour/base of the peaks
    contour_heights = peak_coord_max_2[:,1] - prominences[0]
    
    
    ##############################################################################
    # possible improvement of the choce of the parameter "rel_height" based on how
    # large the base of a peak is. +
    # But her also some problems can accure like in this example:
    # The height goes down to zero and then, the base is maximumized. Here the other
    # shorter height must be take. a function that does this??
    
    
    ##### Calculate information about the widths of the peaks in a given 
    # relative height of the peak set to 0.5
    ##### this gives an estimation of the liners thickness
    peaks_width_info = peak_widths(row_sums_arr,  
                             peak_coord_max_2[:,0],
                             rel_height=0.2,
                             prominence_data = prominences
                             )
        
    peaks_width_info_100 = peak_widths(row_sums_arr,  
                             peak_coord_max_2[:,0],
                             rel_height=1,
                             prominence_data = prominences
                             )
    
    Flute_center_line = (peak_coord_max_2[0,0] + peak_coord_max_2[1,0]) //2
    
    # distance between the approximated center of the liners, which is an 
    # approximation of corrugated board thickness
    dist_liners_center_line = abs(peak_coord_max_2[0,0] - peak_coord_max_2[1,0])
    
    # calculate an estimation of the upper limit of the board 
    board_upper_limit_estimation = upper_liner_center_pos - liners_thickness_estimation//2
    # calculate an estimation of the lower limit of the board 
    board_lower_limit_estimation = lower_liner_center_pos + liners_thickness_estimation//2
    
    # calculate an estimation of the immer limits of the liners 
    upper_liner_inner_limit_estimation = upper_liner_center_pos + liners_thickness_estimation//2
    # calculate an estimation of the lower limit of the board 
    lower_liner_inner_limit_estimation = lower_liner_center_pos - liners_thickness_estimation//2
    
    # estimation of the board thickness
    Board_thickness_estimation = abs(board_upper_limit_estimation - board_lower_limit_estimation)+1

    
    if show_plots:
        # Visualize the output
        plt.figure(figsize=(10, 6))
        plt.plot(row_sums_arr)
        plt.title('Sum of Pixel Values for Each Row')
        plt.xlabel('Row Number')
        plt.ylabel('Sum of Pixel Values')
        
        # plot an x on the 2 highest peaks
        plt.plot(peak_coord_max_2[:,0], peak_coord_max_2[:,1], "x")
        
        # plot vertical lines representing the 2 peaks
        plt.vlines(x = peak_coord_max_2[:,0], 
                   ymax = peak_coord_max_2[:,1], 
                   ymin = np.zeros(len(peak_coord_max_2[:,0])),
                   color='green', linewidth=1)
        
        # plot the peaks prominences
        plt.vlines(x=peak_coord_max_2[:,0], 
                   ymin=contour_heights, 
                   ymax=peak_coord_max_2[:,1],
                   color="C8" , linewidth=2)
        
        # plot horizontal lines representing the widths of peaks
        plt.hlines(*peaks_width_info[1:], color="C3" , linewidth=1 )
        
        # plot horizontal lines representing the widths of peaks
        plt.hlines(*peaks_width_info_100[1:], color="C1" , linewidth=1 )
        
        # zoom in the x axis
        plt.xlim(110,200)
        
        plt.grid("on")
        # show the final plot
        plt.show()
        
        # close the plot
        plt.close()

    return (upper_liner_center_pos,             lower_liner_center_pos,
            board_upper_limit_estimation ,      board_lower_limit_estimation,
            upper_liner_inner_limit_estimation, lower_liner_inner_limit_estimation,
            
            Flute_center_line,
            dist_liners_center_line,
            Board_thickness_estimation)




def column_analysis_fct (horizontal_image,
                         image_upper_lim,
                         image_lower_lim,
                         show_plots = False):
    

    # Initialize a list to save the sum of pixel values for each column
    column_sums_list = []
    
    # Iterate through the columns from the first column to the last column
    for col in range(horizontal_image.shape[1]):
        column_sums_list.append(np.sum(
            horizontal_image[image_upper_lim : image_lower_lim, col]) / (abs(image_upper_lim-image_lower_lim)))
    
    # Convert the list to a NumPy array
    column_sums_arr = np.array(column_sums_list)
    
    # calculate the avrage of the column sums array
    avr_col_sums = np.average(column_sums_arr)
    
    # define a threshold as a percentage of the average value
    threshold_L_R_limits = 0.1
    
    # Find indices where array values intersect the threshold
    possible_L_R_limits = np.where(column_sums_arr >= threshold_L_R_limits * avr_col_sums)[0]
    
    # look for the index of the extreme left and right values over the threshold
    left_board_lim = possible_L_R_limits[0]
    right_board_lim = possible_L_R_limits[-1]
    
    # estimation of the total length of the board in pixels
    # +1 because the limits are part of the board
    board_length_approx = abs(right_board_lim - left_board_lim) +1 
    
    
    if show_plots:
        
        # Visualize the output
        plt.figure(figsize=(10, 6))
        plt.plot(column_sums_arr)
        plt.title('Sum of Pixel Values for Each Column (Left to Right)')
        plt.xlabel('Column Number (from left to right)')
        plt.ylabel('Sum of Pixel Values')
        plt.grid("on")
        
        # plot the horizantol average line
        plt.hlines(y = threshold_L_R_limits * np.average(column_sums_arr), 
                    xmin = 0, 
                    xmax = horizontal_image.shape[1],
                    color='red', linewidth=1)
        
        # plot vertical lines representing the left and right limits
        plt.vlines(x = left_board_lim, 
                    ymin = 0, 
                    ymax = max(column_sums_arr),
                    color='green', linewidth=1)
        plt.vlines(x = right_board_lim, 
                    ymin = 0,
                    ymax = max(column_sums_arr),
                    color='green', linewidth=1)
        
        # show the plot
        plt.show()
        
        # close the plot
        plt.close()

    return left_board_lim, right_board_lim, board_length_approx


def trim_L_R_limits(horizontal_image, trim_size, show_plots = False):
    """
    create board limits to the right and left as if we see the hole sample, 
    by setting some columns to zeros

    Parameters
    ----------
    horizontal_image : TYPE
        DESCRIPTION.
    trim_size : TYPE
        DESCRIPTION.
    show_plots : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    horizontal_image : TYPE
        DESCRIPTION.

    """
    
    horizontal_image[:, :trim_size ] = 0
    horizontal_image[:, -trim_size:] = 0

    if show_plots:
        
        # Show the masked image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(horizontal_image, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title('masked image from the left and right sides')
        plt.show()
        plt.close()

    return horizontal_image


def display_masks_rgb_frame(horizontal_image, 
                              show_plots = False,
                              zoom_in = False,
                              frame_number = None):
    
    """
    Display the calculated limits with colored rectangles and lines: 
        # image-workspace, 
        # the corrugated board limits
        # liners centerlines
        # flute centerline

    Parameters
    ----------
    show_plots : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    # convert the image to rgb to display the limits in colors
    rgb_image = cv2.cvtColor(horizontal_image ,cv2.COLOR_GRAY2RGB)
    
    # Define the top-left and bottom-right corners of the aria image workspace
    top_left = (left_board_lim , image_upper_lim)
    bottom_right = (right_board_lim, image_lower_lim)
    # Define the color (BGR format) and thickness of the rectangle
    color = (0, 255, 0)  # Green color in RGB
    thickness = 2  # Thickness of 2 px
    # Draw the rectangle
    cv2.rectangle(rgb_image, top_left, bottom_right, color, thickness)
    
    
    # Define the top-left and bottom-right corners of the area limited by the board
    top_left = (left_board_lim , board_upper_limit_estimation-1)

    # Use the same effective lower limit as in mask_filter_cleaning
    effective_lower = lower_liner_inner_limit_estimation + LOWER_LINER_MARGIN_PX
    effective_lower = min(effective_lower, board_lower_limit_estimation)
    effective_lower = max(effective_lower, board_upper_limit_estimation + 5)

    bottom_right = (right_board_lim, effective_lower+1)
    # Define the color (BGR format) and thickness of the rectangle
    color = (255, 0, 0)  # Red color in RGB
    thickness = 2  # Thickness of 2 px
    # Draw the rectangle
    cv2.rectangle(rgb_image, top_left, bottom_right, color, thickness)
    
    
    # Define the top-left and bottom-right corners of the aria limited by the liners center
    top_left = (left_board_lim , upper_liner_center_pos)
    bottom_right = (right_board_lim, lower_liner_center_pos)
    # Define the color (BGR format) and thickness of the rectangle
    color = (0, 0, 255)  # Blue color in RGB
    thickness = 2  # Thickness of 2 px
    # Draw the rectangle
    # cv2.rectangle(rgb_image, top_left, bottom_right, color, thickness)
    
    
    # Define the starting and ending points of the horizontal flute center line
    start_point = ( left_board_lim, Flute_center_line)  # Start from the middle left
    end_point = ( right_board_lim, Flute_center_line)  # End at the middle right
    # Define the color (BGR format) and thickness of the line
    color = (0, 0, 255)  # Blue color in RGB
    thickness = 2  # Thickness of the line
    # Draw the horizontal line
    cv2.line(rgb_image, start_point, end_point, color, thickness)
    
    
    ### Zoom in 
    
    x_start_value = 0
    x_end_value = x_start_value+ 200
    x_coord_board = [x_start_value , x_end_value +1]
    
    # Define the region of interest (ROI)
    # (x, y, width, height) in pixels
    x = left_board_lim + x_coord_board[0]
    y = image_upper_lim
    w = x_coord_board[1] - x_coord_board[0]
    h = abs( image_lower_lim - image_upper_lim ) +1
    
    # Crop the image
    cropped_image = rgb_image[ y:y+h ,  x:x+w]
    
    
    if show_plots and not zoom_in:
        
        # Show the original image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(rgb_image, vmin=0, vmax=255)
        plt.axis('on')
        # plt.title(f'Corrugated board limits detection in frame {frame_number}')
        plt.xlabel("Columns in pixels")
        plt.ylabel("Rows in pixels")
        # show the plot
        plt.show()
        # close the plot
        plt.close()
        
    elif show_plots and zoom_in:
        
        # Show the masked image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(cropped_image, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'Corrugated board limits - zoom in frame {frame_number}')
        plt.show()


def mask_filter_cleaning(img, 
                         show_plots = False,
                         show_masked_img = False,
                         show_filtered_img = True,
                         frame_number = 0):
    """
    Masks around the total board and cleaning of the environment
    then filtering process and binarisation
    then liners improvement

    Parameters
    ----------
    horizontal_image : TYPE
        DESCRIPTION.
    show_plots : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    filtered_image : TYPE
        DESCRIPTION.

    """

    # Create a black mask
    mask_clear_workspace = np.zeros(img.shape, dtype=np.uint8)

    # --- NEW: use the inner side of the lower liner as the bottom of the mask ---
    # This keeps the bottom liner but removes the lower steel plate line.
    effective_lower = lower_liner_inner_limit_estimation + LOWER_LINER_MARGIN_PX

    # Safety: do not go below the original board lower limit,
    # and do not cross the upper limit.
    effective_lower = min(effective_lower, board_lower_limit_estimation)
    effective_lower = max(effective_lower, board_upper_limit_estimation + 5)

    mask_clear_workspace[board_upper_limit_estimation-1 : effective_lower+1,
                         left_board_lim-1 : right_board_lim+1] = 255
    # Apply the mask to the horizontal image using a bitwise AND operation
    masked_horizontal_img = cv2.bitwise_and(img, mask_clear_workspace)
    
    
    # apply the filtering process with specific inputs on the masked image
    filtered_img = image_filtering(masked_horizontal_img)
    
    # --- Liner enhancement: brighten top and bottom liner bands ---
    # We use the global geometric limits computed on frame 0:
    #  - board_upper_limit_estimation, board_lower_limit_estimation
    #  - upper_liner_inner_limit_estimation, lower_liner_inner_limit_estimation
    #  - left_board_lim, right_board_lim

    try:
        # Top liner band
        filtered_img[board_upper_limit_estimation:upper_liner_inner_limit_estimation,
                     left_board_lim:right_board_lim] = 255

        # Bottom liner band
        filtered_img[lower_liner_inner_limit_estimation:board_lower_limit_estimation,
                     left_board_lim:right_board_lim] = 255
    except NameError:
        # If for some reason these globals are not available, just skip enhancement
        pass
    
    # # liners improvement
    # # the liners set to the maximum value 255
    # filtered_img[board_upper_limit_estimation : upper_liner_inner_limit_estimation , 
    #                 left_board_lim : right_board_lim] = 255
    
    # filtered_img[lower_liner_inner_limit_estimation : board_lower_limit_estimation , 
    #                 left_board_lim : right_board_lim] = 255
    
    
    # liners improvement
    # the liners set to the maximum value 255
    filtered_img[   upper_liner_center_pos , 
                    left_board_lim : right_board_lim] = 255
    
    filtered_img[   lower_liner_center_pos , 
                    left_board_lim : right_board_lim] = 255
    
    
    if show_plots and show_masked_img:
        
        # Show the masked image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(masked_horizontal_img, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'masked and cleaned frame {frame_number}')
        plt.show()

    if show_plots and show_filtered_img:

        # Show the masked image
        plt.figure(figsize= (fig_size, fig_size))
        plt.imshow(filtered_img, "gray", vmin=0, vmax=255)
        plt.axis('off')
        plt.title(f'masked, cleaned and filtered frame {frame_number}')
        plt.show()
        
        
    
    return filtered_img



if new_image_analysis: 
    
    # define the size of figures
    fig_size = 15
    
    # Define the paths to the folder of frames, 
    # read the first frame, 
    
    # define the path to the first frame (starting by frame 0 or 1)
    # Frame 0 is a key frame: it is different from all the other frames that were compressed
    # This difference results into false node displacements between frame 0 and 1.
    

    # list all image files in frames_folder
    files = [f for f in os.listdir(frames_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    files.sort()
    
    if not files:
        raise RuntimeError(f"No image files found in {frames_folder}")
    
    first_img_path = os.path.join(frames_folder, files[0])
    gray_image = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
    
    if gray_image is None:
        raise RuntimeError(f"Could not read first frame: {first_img_path}")
    
    # display Histogram to set a threshold
    # # visualize the Histogram of the gray image to manually choose a threshold
    # pd.Series(gray_image.flatten()).plot(kind='hist',
    #                                          bins=60,
    #                                          title='Distribution of Pixel Values',
    #                                          ylim=(0,2500))
    
    # Plot the original image
    plt.figure(figsize= (fig_size, fig_size))
    plt.imshow(gray_image, "gray", vmin=0, vmax=255)
    plt.axis('off')
    plt.title('original image')
    plt.show()
    
    
    # rough row analysis of the input image
    (image_upper_lim_rough,
     image_lower_lim_rough) = rough_workspace_limits(gray_image)
    
    # rotation angle stuff ...
    
    # rotate the first frame
    horizontal_image = rotate(gray_image, ROTATION_ANGLE, resize=False)
    horizontal_image = (horizontal_image * 255).astype(np.uint8)
    
    # trim left/right
    trim_size = 6
    horizontal_image = trim_L_R_limits(horizontal_image, trim_size)
    
    # --- OPTIONAL: see the original "Averaged value..." plot BEFORE correction ---
    # row_analysis_fct_frame_0(horizontal_image,
    #                          image_upper_lim_rough,
    #                          image_lower_lim_rough,
    #                          show_plots=True)
    
    # --- NEW: correct the vertical shadow between upper and lower rough limits ---
    horizontal_image = correct_vertical_shadow(horizontal_image,
                                               image_upper_lim_rough,
                                               image_lower_lim_rough,
                                               smooth_sigma=3,
                                               shadow_factor=0.6,
                                               max_gain=3.0,
                                               show_plots=True)  # <- row-mean before/after
    
    # Now do the usual row analysis on the CORRECTED image
    (upper_liner_center_pos,            lower_liner_center_pos,
     board_upper_limit_estimation ,     board_lower_limit_estimation,
     upper_liner_inner_limit_estimation, lower_liner_inner_limit_estimation,
     image_upper_lim,                    image_lower_lim,
     liners_thickness_estimation,
     Flute_center_line,
     dist_liners_center_line,
     Board_thickness_estimation) = row_analysis_fct_frame_0(horizontal_image,
                                                            image_upper_lim_rough,
                                                            image_lower_lim_rough,
                                                            show_plots=True)
                                
    # column analysis of the horizontal frame 0
    # To set the right and left limits of the corrugated board
    # for next frames, these same limits are kept
    
    (left_board_lim, 
     right_board_lim, 
     board_length_approx)   =   column_analysis_fct(horizontal_image,
                                                    image_upper_lim,
                                                    image_lower_lim,
                                                    show_plots = False)
    
                                                    
    # Display the calculated limits with colored rectangles and lines: 
        # image-workspace, 
        # the corrugated board limits
        # liners centerlines
        # flute centerline
    display_masks_rgb_frame(horizontal_image, 
                              show_plots = True, 
                              zoom_in= False,
                              frame_number= 0 )
    
    # Masks around the total board and cleaning of the environment, filtering
    # correct the liners missing
    filtered_image = mask_filter_cleaning(horizontal_image, show_plots = True)
    
    
    
    ###############################################################################
    # Functions for graph analysis and plotting of graphs
    
    def remove_degree_one_nodes(G):
        """
        Safer version:
        - Remove nodes that have degree 1 in a *single* pass.
        - Do NOT loop until the graph is empty.
    
        This trims tiny dangling branches but preserves all main chains/loops.
        """
        nodes_to_remove = [node for node, deg in G.degree() if deg == 1]
        G.remove_nodes_from(nodes_to_remove)
            
     # Function to remove all nodes that are not on the top or bottom liner rows
    def remove_middle_nodes_keep_liners(G,
                                        upper_row,
                                        lower_row,
                                        liner_thickness_px=3,
                                        extra_margin_px=0):
        """
        Keep only nodes that lie on (or very close to) the upper and lower liner
        center rows. All other 'middle' nodes are removed.

        Nodes with degree 2 are removed with reconnection of their neighbours,
        using the same logic as in remove_degree_one_nodes.
        """
        # Vertical tolerance so we don't lose liner nodes due to 1–2 px noise
        tol = max(1, int(liner_thickness_px // 2) + extra_margin_px)

        # Identify which nodes we want to keep (top and bottom rows)
        nodes_to_keep = set()
        for node, data in G.nodes(data=True):
            r, c = data['o']  # (row, col)
            if abs(r - upper_row) <= tol or abs(r - lower_row) <= tol:
                nodes_to_keep.add(node)

        # Everything else = middle nodes that we want to remove
        nodes_to_remove = [n for n in G.nodes() if n not in nodes_to_keep]

        print(f"[middle-node cleanup] nodes before: {G.number_of_nodes()}, "
              f"to remove (middle): {len(nodes_to_remove)}")

        # Remove nodes one by one and reconnect neighbours if they have degree 2
        for n in nodes_to_remove:
            neighbours = list(G.neighbors(n))

            # Same reconnection rule as in remove_degree_one_nodes
            if len(neighbours) == 2:
                n1, n2 = neighbours
                if not G.has_edge(n1, n2):
                    y1, x1 = G.nodes[n1]['o']
                    y2, x2 = G.nodes[n2]['o']
                    pts = np.array([[y1, x1],
                                    [y2, x2]], dtype=np.int32)
                    G.add_edge(n1, n2, pts=pts)

            # Finally remove the node
            G.remove_node(n)

        print(f"[middle-node cleanup] nodes after: {G.number_of_nodes()}")
                
    
    def graph_draw_fct (image, graph, frame_number = None,
                        node_pos_dict = None,
                        markersize= 12, fontsize= 20, title = None,
                        node_labels = True,
                        zoom_in = False,
                        x_min = left_board_lim,
                        x_max = 200,
                        white_background = False,
                        nx_direct_edges = False,
                        save_figures = False,
                        destination_folder_graphs = 'graph_segmentation_figs' ):
        
    
        if  not nx_direct_edges :
        
            if node_labels and not zoom_in:
                
                plt.figure(figsize=(fig_size , fig_size ))
                # draw image
                plt.imshow(image, cmap='gray')
                plt.axis('on')
                
                # draw edges by pts
                for (s,e) in graph.edges():
                    ps = graph[s][e]['pts']
                    plt.plot(ps[:,1], ps[:,0], 'green')
                
                # nodes must be updated in a global varibale 
                # (not locally in a function) nodes = graph.nodes()
                for i in graph.nodes():
                    node = graph.nodes[i]
                    y, x = node['o']
                    plt.plot(x, y, 'ro', markersize = markersize)
                    plt.text(x, y, str(i), color='white', fontsize = fontsize, ha='center',
                             va='center', fontweight='bold')
                # title and show
                plt.title(title)
                # plt.show()
            
            
            # plot the zoomed region
            elif node_labels and zoom_in:
        
                
                # Define the zoomed-in region
                y_min, y_max = image_upper_lim, image_lower_lim  # Y-axis limits
                
                # Calculate the aspect ratio of the zoomed-in region
                aspect_ratio = (x_max - x_min) / (y_max - y_min)
                
                # Define the figure size, maintaining the aspect ratio
                fig_width = fig_size  # Adjust as needed for larger plot
                fig_height = fig_width / aspect_ratio
                
                # Create a new figure with the calculated size
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # Draw the image
                ax.imshow(image, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
                
                # Set axis limits to zoom into the specified region
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # Invert the y-axis to match image coordinates
                
                # Adjust the aspect ratio to maintain the original image's aspect ratio
                ax.set_aspect(aspect=1)
                
                # Enable the axis
                ax.axis('on')
                
                # Create a subgraph with only the nodes and edges in the specified region
                subgraph_nodes = [n for n in graph.nodes()]
                nodes = graph.nodes()
                subgraph = graph.subgraph(subgraph_nodes)
                
                # Draw edges by pts
                for (s, e) in subgraph.edges():
                    print(graph[s][e])
                    ps = graph[s][e]['pts']
                    ax.plot(ps[:, 1], ps[:, 0], 'green')
                
                # Draw nodes
                for i in subgraph.nodes():
                    node = nodes[i]
                    y, x = node['o']
                    ax.plot(x, y, 'ro', markersize= markersize)  # Adjust markersize as needed
                    ax.text(x, y, str(i), color='white', fontsize= fontsize , ha='center', va='center', fontweight='bold')  # Adjust fontsize as needed
        
                # title and show
                plt.title(title)
                
                # show the plot
                # plt.show()
                
                
            elif not node_labels and not zoom_in:
                ######################################### draw without node numbers
                # initiate a figure
                plt.figure(figsize= (fig_size, fig_size))
                plt.axis('off')
                # Print the skeleton
                plt.imshow(ske_thin, "gray", vmin=0, vmax=255)
                # draw edges by pts
                for (s,e) in graph.edges():
                    print(graph[s][e])
                    if graph[s][e]:
                        ps = graph[s][e]['pts']
                        plt.plot(ps[:,1], ps[:,0], 'green')
                # draw node by o
                nodes = graph.nodes()
                ps = np.array([nodes[i]['o'] for i in nodes])
                plt.plot(ps[:,1], ps[:,0], 'r.',markersize = 5)
                # title and show
                plt.title(title)
                # plt.show()
                
            else:
                
                ######################################### draw without node numbers and zoom in
                # initiate a figure
                plt.figure(figsize= (fig_size, fig_size))
                plt.axis('off')
                
                
                # Print the skeleton
                plt.imshow(ske_thin, "gray", vmin=0, vmax=255)
                # draw edges by pts
                for (s,e) in graph.edges():
                    ps = graph[s][e]['pts']
                    plt.plot(ps[:,1], ps[:,0], 'green')
                # draw node by o
                ps = np.array([nodes[i]['o'] for i in nodes])
                plt.plot(ps[:,1], ps[:,0], 'r.',markersize = markersize)
        
                # title and show
                plt.title(title)
                # plt.show()
                
            
        elif nx_direct_edges and not white_background :
            
            if not zoom_in :
            
                # Print the binary image
                plt.figure(figsize= (fig_size, fig_size))
                plt.imshow(ske_thin, "gray", vmin=0, vmax=255)
        
                # draw the graph
                nx.draw(graph, node_pos_dict , with_labels=False,
                        node_color="red",node_size=markersize, edge_color="green",
                        font_color="white",font_size=10, 
                        font_weight="bold", width=2)
                plt.title(title)
        
                # title and show
                # plt.title(title)
                # plt.show()
            
            elif  zoom_in:
            
                # Define the zoomed-in region
                y_min, y_max = image_upper_lim, image_lower_lim  # Y-axis limits
                
                # Calculate the aspect ratio of the zoomed-in region
                aspect_ratio = (x_max - x_min) / (y_max - y_min)
                
                # Define the figure size, maintaining the aspect ratio
                fig_width = fig_size  # Adjust as needed for larger plot
                fig_height = fig_width / aspect_ratio
                
                # Create a new figure with the calculated size
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                ax.set_axis_off()
                
                # Draw the image
                ax.imshow(image, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
                
                # Set axis limits to zoom into the specified region
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # Invert the y-axis to match image coordinates
                
                # Adjust the aspect ratio to maintain the original image's aspect ratio
                ax.set_aspect(aspect=1)
                
                # Enable the axis
                ax.axis('on')
                
                # Create a subgraph with only the nodes and edges in the specified region
                subgraph_nodes = [n for n in graph.nodes() if x_min-15 <= nodes[n]['o'][1] <= x_max+15 and y_min <= nodes[n]['o'][0] <= y_max]
                subgraph = graph.subgraph(subgraph_nodes)
                
        
                # draw the graph
                nx.draw(graph, node_pos_dict , with_labels=True,
                        node_color="red",node_size=500, edge_color="green",
                        font_color="white",font_size=30, 
                        font_weight="bold", width=6)
                
                
                # title and show
                plt.title(title)
                
                # show the plot
                # plt.show()
                
        
        if  white_background and not zoom_in and not nx_direct_edges :
            
            # make a white background
            white_img = (255 * np.ones(image.shape))
            # Show the masked image
            plt.figure(figsize= (fig_size, fig_size))
            # print the white background
            plt.imshow(white_img, "gray", vmin=0, vmax=255)
            plt.axis('off')
            
            
            # draw edges by pts
            for (s,e) in graph.edges():
                ps = graph[s][e]['pts']
                plt.plot(ps[:,1], ps[:,0], 'green')
            # draw node by o
            ps = np.array([nodes[i]['o'] for i in nodes])
            plt.plot(ps[:,1], ps[:,0], 'r.',markersize = 5)
            
            
            plt.title(title)
            # plt.show()
            
        elif white_background and not zoom_in and nx_direct_edges :
        
            # make a white background
            white_img = (255 * np.ones(image.shape))
            # Show the masked image
            plt.figure(figsize= (fig_size, fig_size))
            # print the white background
            plt.imshow(white_img, "gray", vmin=0, vmax=255)
            plt.axis('off')
    
            # draw the graph
            nx.draw(graph, node_pos_dict , with_labels=True,
                    node_color="red",node_size=10, edge_color="green",
                    font_color="white",font_size=0, 
                    font_weight="bold", width=1)
            
            
            plt.title(title)
            # plt.show()
            
            
        if save_figures:
            # Create a directory to store the image if it doesn't exist
            if destination_folder_graphs is not None:
                os.makedirs(destination_folder_graphs, exist_ok=True)
                image_path = os.path.join(
                    destination_folder_graphs,
                    f"Graph_Frame_{frame_number:04d}.png"   # <-- changed to png
                )
                plt.savefig(image_path, bbox_inches='tight', dpi=300)
                print(f"[INFO] Saved graph figure to: {image_path}")

    
            # # Save the plot as an image in the destination directory
            image_path = os.path.join(destination_folder_graphs, f'Graph_Frame_{frame_number:04d}.jpg')
            
            # plt.savefig(image_path)
            plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    
            plt.show()
            plt.close()
        
        else:
            plt.show()
            plt.close()
    
            
                
            
            
   ############## Skeletonize and build a graph from the filtered frame 0

    # make a skeleton through skeletonization or thinning of the structure
    ske_thin = thin(filtered_image).astype(np.uint8)*255
    ske_skeletonize = skeletonize(filtered_image).astype(np.uint8)*255
    
    # --- NEW: remove the bottom plate line from the skeleton ---
    # Label connected components on the skeleton
    labels = label(ske_skeletonize > 0)
    props = regionprops(labels)
    
    plate_label = None
    best_row = -1
    
    height, width = ske_skeletonize.shape
    # parameters: "long & thin" = horizontal line
    MIN_WIDTH = int(0.6 * width)   # must span at least 60% of the width
    MAX_HEIGHT = 5                 # at most 5 pixels tall
    
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        comp_height = maxr - minr
        comp_width = maxc - minc
    
        # keep only long, thin components (horizontal-ish lines)
        if comp_width < MIN_WIDTH:
            continue
        if comp_height > MAX_HEIGHT:
            continue
    
        # pick the lowest such component (bottom plate line)
        if maxr > best_row:
            best_row = maxr
            plate_label = p.label
    
    if plate_label is not None:
        plate_mask = (labels == plate_label)
        # remove that component from both skeleton images
        ske_skeletonize[plate_mask] = 0
        ske_thin[plate_mask] = 0
        print(f"Removed bottom plate component (label={plate_label}, max_row={best_row})")
    else:
        print("No clear bottom plate component found in skeleton.")
    
    # -------------------------------------------------------------
    
    # 1) build graph
    if Ske_thin_method:
        graph_0 = sknw.build_sknw(ske_skeletonize)
    else:
        graph_0 = sknw.build_sknw(ske_thin)
    
    print("[DEBUG] frame 0 graph:", graph_0.number_of_nodes(), "nodes,",
      graph_0.number_of_edges(), "edges")

    simplify_small_cycles_edge_based(
        graph_0,
        small_area_ratio=0.25,
        big_area_ratio=0.7,
        max_iter=30,
        show_hist_initial=True,
        show_hist_each_iter=True,   # <- histogram every iteration
        big_growth_tol=1.5,
    )
    
    # then compress/prune degrees as discussed earlier:
    graph_0 = compress_degree2_paths(graph_0, border_x_tol=15)
    prune_degrees(graph_0, border_x_tol=15)
    
    print("[DEBUG] after all simplifications:", graph_0.number_of_nodes(),
          "nodes,", graph_0.number_of_edges(), "edges")
    
    # 4) now do your renaming + node_pos_dict + final plotting
    graph_draw_fct(
        horizontal_image,
        graph_0,
        title="Graph after all simplifications (frame 0)",
        node_labels=True,
        zoom_in=True,
        x_max=500,
    )



    # draw graph with nodes
    graph_draw_fct (ske_skeletonize, graph_0, 
                    title = "Skeleton graph from frame number 0")
    
    # draw graph with nodes
    graph_draw_fct (horizontal_image, graph_0,
                    title = "Skeleton graph from frame number 0",
                    node_labels = True,
                    zoom_in = True,
                    x_max = 500)
    
    
    
    # call function to remove nodes with degree 1 
    #remove_degree_one_nodes(graph_0)
    
    #remove_middle_nodes_keep_liners(
    #    graph_0,
    #    upper_liner_center_pos,
    #    lower_liner_center_pos,
    #    liner_thickness_px=int(max(1, round(liners_thickness_estimation))),
    #    extra_margin_px=0,   # you can set this to 1 or 2 if needed
    #)

    # Optional: visualize cleaned graph for frame 0
    graph_draw_fct(
        horizontal_image,
        graph_0,
        title="Graph after removing middle nodes (frame 0)",
        node_labels=True,
        zoom_in=True,
        x_max=500
    )
    
    print("[DEBUG] frame 0 graph:", graph_0.number_of_nodes(), "nodes,",
      graph_0.number_of_edges(), "edges")
    
    print("[DEBUG] before triangle + deg1 cleanup:",
      graph_0.number_of_nodes(), "nodes,",
      graph_0.number_of_edges(), "edges")

    # 1) remove triangles that are pure (edges not used by bigger cycles)
    remove_pure_triangles(graph_0, max_iter=20)
    
    # 2) remove all degree-1 nodes (safe: they are not part of cycles)
    remove_all_degree_one_nodes(graph_0)
    
    print("[DEBUG] after triangle + deg1 cleanup:",
          graph_0.number_of_nodes(), "nodes,",
          graph_0.number_of_edges(), "edges")
    
    # now recompute nodes/edges for the renaming block
    nodes = graph_0.nodes()
    edges = graph_0.edges()

    # initiate an array of zeros ...
    nodes_coord_array = np.zeros( (graph_0.number_of_nodes(),3) ).astype(int)
    counter = 0
        
    # for loop to read and store the node information in the array as follows:
    # every row containes: [node_name , node_row_position , nodes_column_position]
    for node in nodes:
        
        node_info_list = [node, nodes[node]['o'][0] , nodes[node]['o'][1]]
        nodes_coord_array[counter,:] = np.array(node_info_list)
        # increment the counter
        counter +=1
        
        
    # Sort the array based on the column position of the nodes from left to right
    nodes_coord_array_sorted = nodes_coord_array[nodes_coord_array[:, 2].argsort()]
    
    # initiate a dictionary to associate new names to every nodes
    nodes_new_names_dict = {}
    
    # fill the dictionary with the node names as keys and new name as values
    for new_name, old_name in enumerate(nodes_coord_array_sorted[:,0]) :
        nodes_new_names_dict[old_name] = new_name
        
    # rename the nodes based on the dictionary created
    graph_0 = nx.relabel_nodes(graph_0, nodes_new_names_dict)
    
    
    # update the variables nodes and edges (we can print nodes before and 
    # after updating to see the difference)
    # print(nodes)
    nodes = graph_0.nodes()
    # print(nodes)
    edges = graph_0.edges()
    
    nodes_number_list = []
    nodes_number_list.append(graph_0.number_of_nodes())
    
    # update the previous array fo the next frame 
    # and with 3 columns for [node_name , node_row_position , nodes_column_position]
    nodes_coord_array = np.zeros( (graph_0.number_of_nodes() + 10 , 3) ).astype(int)
    # initiate a counter of the node number to 0
    counter = 0
    # for loop to read and store the node information in the array as follows:
    # every row containes: [node_name , node_row_position , nodes_column_position]
    for node in nodes:
        
        node_info_list = [node, nodes[node]['o'][0] , nodes[node]['o'][1]]
        nodes_coord_array[counter,:] = np.array(node_info_list)
        if nodes_coord_array[counter,1] == 0 and nodes_coord_array[counter,2] == 0:
            nodes_coord_array[counter, 0] = -1
        # increment the counter
        counter +=1
    
    for i in range(counter, nodes_coord_array.shape[0]):
        if nodes_coord_array[i, 1] == 0 and nodes_coord_array[i, 2] == 0:
            nodes_coord_array[i, 0] = -1
    
    # # store the nodes infos
    # nodes_list_pre = list(graph_0.nodes)
    # nodes_dict_pre = dict(graph_0.nodes)
    # edges_list_pre = list(graph_0.edges)
    # edges_dict_pre = dict(graph_0.edges)
    
    
    filtered_arr = nodes_coord_array[nodes_coord_array[:, 0] != -1]
    # Convert ndarray to DataFrame
    df_node_pos = pd.DataFrame(filtered_arr)
    
    # Set the first column as the index
    df_node_pos.set_index(0, inplace=True)
    
    node_pos_dict = {index: (row[2], row[1]) for index, row in df_node_pos.iterrows()}
    
    # -------------------------------------------------------------
    # NEW: save absolute pixel coordinates for the initial frame
    #       (node_id, row_px, col_px)
    # -------------------------------------------------------------
    coords_df = df_node_pos.copy()
    coords_df.columns = ["row_px", "col_px"]  # col 1 = row, col 2 = col (pixels)
    
    coords_csv_path = os.path.join(
        destination_folder_graphs,
        "node_coordinates_frame0_pixels.csv"
    )
    coords_df.to_csv(coords_csv_path, index_label="node_id")
    print(f"Saved initial node pixel coordinates to {coords_csv_path}")

    
    # draw the filtered graph with relabelled nodes (on top of the image)
    graph_draw_fct(
        horizontal_image,
        graph_0,
        frame_number=0,
        node_pos_dict=node_pos_dict,
        title="Graph, frame number 0",
        node_labels=True,
        zoom_in=False,
        x_max=1020,
        nx_direct_edges=True,
        save_figures=save_figures_global,
        destination_folder_graphs=destination_folder_graphs,
    )
    
    # draw graph alone on white background
    graph_draw_fct(
        horizontal_image,
        graph_0,
        frame_number=0,
        node_pos_dict=node_pos_dict,
        title="Graph, frame number 0",
        node_labels=False,
        zoom_in=False,
        x_max=200,
        white_background=True,
        nx_direct_edges=True,
        save_figures=save_figures_global,
        destination_folder_graphs=destination_folder_graphs,
    )

    
    # draw the filtered graph with relablled nodes
    graph_draw_fct (horizontal_image, graph_0, frame_number = 0,
                    node_pos_dict = node_pos_dict,
                    title = "Graph, frame number 0",
                    node_labels = True,
                    zoom_in = False,
                    x_max = 1020,
                    nx_direct_edges = True,
                    save_figures = save_figures_global,
                    destination_folder_graphs = destination_folder_graphs)

    
    
    
    # draw graph with nodes
    graph_draw_fct (horizontal_image, graph_0, frame_number = 0,
                    node_pos_dict = node_pos_dict,
                    title = "Graph, frame number 0",
                    node_labels = False,
                    zoom_in = False,
                    x_max = 200,
                    white_background = True,
                    nx_direct_edges = True)
    

        
    
    Graphs_List = []
    Graphs_List.append(graph_0)
    
    ######### Idea to filter the graphs
    #   2 Groups of nodes   upper liner nodes
    #                       lower liner nodes
    # Cosdition: every node must have   2 edges with nodes from his group
    #                                   1 edges with nodes from the other group
    #########
    
    ##############################################################################
    # Quick visual: 4 representative frames (make sure indices exist)
    files_preview = [f for f in os.listdir(frames_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg',
                                            '.bmp', '.tif', '.tiff'))]
    files_preview.sort()
    num_preview = len(files_preview)
    
    if num_preview > 0:
        # choose up to 4 roughly equidistant frames
        idxs = np.linspace(0, num_preview - 1, 4, dtype=int)
        frames_to_plot = idxs.tolist()
    
        titles_list = ['Initial state',
                       'Early compression',
                       'Middle of compression',
                       'Late compression'][:len(frames_to_plot)]
    
        fig_master, axs_master = plt.subplots(len(frames_to_plot), 1,
                                              figsize=(10, 10))
        # when there is only one subplot, axs_master is not a list
        if len(frames_to_plot) == 1:
            axs_master = [axs_master]
    
        for ax, frame_idx, title in zip(axs_master, frames_to_plot, titles_list):
            current_img_path = os.path.join(frames_folder, files_preview[frame_idx])
            current_gray_img = cv2.imread(current_img_path,
                                          cv2.IMREAD_GRAYSCALE)
    
            if current_gray_img is None:
                print(f"[WARN] Could not read {current_img_path}, skipping preview.")
                ax.axis('off')
                continue
    
            ax.imshow(current_gray_img, cmap='gray')
            ax.set_title(title, fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=8)
    
        plt.tight_layout()
        plt.show()
    ##############################################################################
   
    
    
    # Get the list of files in the folder containing the frames
    files = [f for f in os.listdir(frames_folder)
         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    files.sort()
    
    num_frames = len(files)
    
    # initialisation of an empty dictionary for the node tracking
    node_track_dict_dist = {}
    
    thickness_0 = Board_thickness_estimation
    thickness_list = [thickness_0]
    
    compression_phase = False
    
    alpha_average_dict = {}
    alpha_average_list = []
    alpha_average_list_2 = []
    alpha_variance_list = []
    alpha_variance_list_2 = []
    division_by_0 = []
    alpha_list_master = [[] for i in range( 0 , len(files))]
    alpha_list_master_2 = [False for i in range(50)]
    count_list = []
    count = 0
    
    
    for node in range(-1, 60):
        node_track_dict_dist[node] = []
    
    
    
    for frame_number in range(1, num_frames):
    
        # ----------------------------------------------------------
        # 0) Load and preprocess image for current frame
        # ----------------------------------------------------------
        current_img_path = os.path.join(frames_folder, files[frame_number])
        current_gray_img = cv2.imread(current_img_path, cv2.IMREAD_GRAYSCALE)
        if current_gray_img is None:
            raise RuntimeError(f"Could not read frame: {current_img_path}")
        
        # rotate the image
        current_horizontal_image = rotate(current_gray_img, ROTATION_ANGLE,
                                          resize=False)
        current_horizontal_image = (current_horizontal_image * 255).astype(np.uint8)
        
        # apply same lateral trim as frame 0
        current_horizontal_image = trim_L_R_limits(current_horizontal_image,
                                                   trim_size)
        
        # equalise vertical shadow in exactly the same way as for frame 0
        current_horizontal_image = correct_vertical_shadow(
            current_horizontal_image,
            image_upper_lim_rough,
            image_lower_lim_rough,
            smooth_sigma=3,
            shadow_factor=0.5,
            max_gain=7.0,
            show_plots=False
        )
    
        # ----------------------------------------------------------
        # 1) Row analysis for this frame
        # ----------------------------------------------------------
        (upper_liner_center_pos, lower_liner_center_pos,
         board_upper_limit_estimation, board_lower_limit_estimation,
         upper_liner_inner_limit_estimation,
         lower_liner_inner_limit_estimation,
         Flute_center_line,
         dist_liners_center_line,
         Board_thickness_estimation) = row_analysis_fct_next_frame(
             current_horizontal_image,
             image_upper_lim,
             image_lower_lim,
             liners_thickness_estimation,
             show_plots=False
        )
    
        thickness_list.append(Board_thickness_estimation)
    
        display_masks_rgb_frame(current_horizontal_image, 
                                show_plots=False, 
                                zoom_in=False,
                                frame_number=frame_number)
    
        # ----------------------------------------------------------
        # 2) Masking, cleaning, skeletonisation
        # ----------------------------------------------------------
        current_filtered_image = mask_filter_cleaning(
            current_horizontal_image, 
            show_plots=False,
            show_masked_img=True,
            show_filtered_img=True,
            frame_number=frame_number
        )
    
        ske_thin = thin(current_filtered_image).astype(np.uint8) * 255
        ske_skeletonize = skeletonize(current_filtered_image).astype(np.uint8) * 255
        
        if Ske_thin_method:
            raw_graph = sknw.build_sknw(ske_skeletonize)
        else:
            raw_graph = sknw.build_sknw(ske_thin)
    
        # remove degree-1 nodes in the RAW skeleton
        remove_degree_one_nodes(raw_graph)
    
        # ----------------------------------------------------------
        # 3) Build coord array for RAW skeleton of this frame
        #    nodes_coord_array_current: [node_id, row, col]
        # ----------------------------------------------------------
        nodes = raw_graph.nodes()
        nodes_coord_array_current = np.zeros((raw_graph.number_of_nodes() + 10, 3),
                                             dtype=int)
        counter = 0
        for node in nodes:
            r, c = raw_graph.nodes[node]['o']
            nodes_coord_array_current[counter, :] = np.array([node, r, c])
            if r == 0 and c == 0:
                nodes_coord_array_current[counter, 0] = -1
            counter += 1
    
        for i in range(counter, nodes_coord_array_current.shape[0]):
            if (nodes_coord_array_current[i, 1] == 0 and
                    nodes_coord_array_current[i, 2] == 0):
                nodes_coord_array_current[i, 0] = -1
    
        # ----------------------------------------------------------
        # 4) TRACKING: move the nodes of graph_0 only
        # ----------------------------------------------------------
        tracked_graph = graph_0.copy()  # same IDs & edges as frame 0
    
        # next "previous position" array for the NEXT frame
        nodes_coord_array_next = np.zeros_like(nodes_coord_array, dtype=int)
        next_counter = 0
    
        for pre_node, pre_row0, pre_col0 in nodes_coord_array:
            if pre_node == -1:
                continue  # skip padding rows
    
            # previous best coordinate for this node (if we have history)
            history = node_track_dict_dist.get(pre_node, [])
            if history:
                pre_row, pre_col = history[-1][3:5]   # last matched (row, col)
            else:
                pre_row, pre_col = pre_row0, pre_col0
    
            best_dist = np.inf
            best_r, best_c = pre_row, pre_col
    
            # search nearest node in the RAW skeleton
            for c_node, c_row, c_col in nodes_coord_array_current:
                if c_node == -1:
                    continue  # ignore dummy rows
                d = math.hypot(c_row - pre_row, c_col - pre_col)
                if d < best_dist and d <= MAX_TRACK_DIST:
                    best_dist = d
                    best_r, best_c = c_row, c_col
    
            # detect compression start using actually matched position
            if (not compression_phase) and best_dist <= MAX_TRACK_DIST and abs(best_r - pre_row) > 2:
                compression_phase = True
                compression_phase_start_frame = frame_number
    
            # only update if we found a close enough candidate
            if best_dist <= MAX_TRACK_DIST:
                # update coordinates of this node in the tracked graph
                tracked_graph.nodes[pre_node]['o'] = np.array([best_r, best_c],
                                                              dtype=int)
    
                # store tracking history
                node_track_dict_dist.setdefault(pre_node, []).append(
                    (frame_number, pre_row, pre_col,
                     best_r, best_c, float(best_dist))
                )
            else:
                # keep previous coordinates
                tracked_graph.nodes[pre_node]['o'] = np.array([pre_row, pre_col],
                                                              dtype=int)
    
            # fill the "previous" array for next frame
            nodes_coord_array_next[next_counter, :] = np.array(
                [pre_node, tracked_graph.nodes[pre_node]['o'][0],
                          tracked_graph.nodes[pre_node]['o'][1]]
            )
            next_counter += 1
    
        # pad remaining rows with -1 in the id column
        for i in range(next_counter, nodes_coord_array_next.shape[0]):
            nodes_coord_array_next[i, 0] = -1
            nodes_coord_array_next[i, 1] = 0
            nodes_coord_array_next[i, 2] = 0
    
        # update the "previous frame" array
        nodes_coord_array = nodes_coord_array_next
    
        # from now on, use tracked_graph as current_graph
        current_graph = tracked_graph
    
        # book-keeping
        nodes_number_list.append(current_graph.number_of_nodes())
        curr_graph_edge_dict = dict(current_graph.edges())
        curr_graph_node_dict = dict(current_graph.nodes())
    
        # ----------------------------------------------------------
        # 5) Build df_node_pos and node_pos_dict from tracked_graph
        # ----------------------------------------------------------
        filtered_arr = nodes_coord_array[nodes_coord_array[:, 0] != -1]
        df_node_pos = pd.DataFrame(filtered_arr)
        df_node_pos.set_index(0, inplace=True)
    
        node_pos_dict = {
            idx: (row[2], row[1])    # (x = col, y = row)
            for idx, row in df_node_pos.iterrows()
        }
    
        # ensure each edge has a "pts" polyline (straight segment)
        for u, v, data in current_graph.edges(data=True):
            y1, x1 = current_graph.nodes[u]['o']
            y2, x2 = current_graph.nodes[v]['o']
            data['pts'] = np.array([[y1, x1], [y2, x2]], dtype=np.int16)
    
        # save this frame’s tracked graph
        Graphs_List.append(current_graph)
    
        # ----------------------------------------------------------
        # 6) Draw graphs for visual debugging
        # ----------------------------------------------------------
        graph_draw_fct(
            current_horizontal_image,
            current_graph,
            frame_number,
            node_pos_dict,
            title="Graph, frame number " + str(frame_number),
            node_labels=False,
            zoom_in=False,
            nx_direct_edges=False,
            x_max=500,
        )
    
        graph_draw_fct(
            current_horizontal_image,
            current_graph,
            frame_number,
            node_pos_dict,
            title="Graph, frame number " + str(frame_number),
            node_labels=True,
            zoom_in=False,
            x_max=1020,
            nx_direct_edges=True,
            save_figures=save_figures_global,
            destination_folder_graphs=destination_folder_graphs,
        )
    
        # ----------------------------------------------------------
        # 7) Your alpha computations (unchanged, using df_node_pos)
        # ----------------------------------------------------------
        nodes_2_dict = dict(current_graph.nodes())
        edges_2_dict = dict(current_graph.edges())
    
        alpha_list = []
        count_edge = -1
    
        for i in range(0, max(nodes_2_dict.keys()), 4):
    
            if len(alpha_list_master[frame_number]) < 22:
    
                if (i, i+1) in edges_2_dict.keys() or (i+1, i) in edges_2_dict.keys():
    
                    r_0 = df_node_pos.loc[i, 1]
                    c_0 = df_node_pos.loc[i, 2]
    
                    r_1 = df_node_pos.loc[i+1, 1]
                    c_1 = df_node_pos.loc[i+1, 2]
    
                    if (r_1 - r_0) != 0:
                        alpha = math.atan((c_1 - c_0) / (r_1 - r_0)) * 180 / np.pi
                    else:
                        alpha = 90.0
    
                    alpha_list.append(alpha)
                    alpha_list_master[frame_number].append(alpha)
    
                    count_edge += 1
    
                    if frame_number > 1:
                        if (alpha_list_master[frame_number-1][count_edge] < 0 and
                                not alpha_list_master_2[count_edge]):
                            alpha_list_master_2[count_edge] = not alpha_list_master_2[count_edge]
                            count += 1
    
                else:
                    count_edge += 1
                    alpha_list_master[frame_number].append(
                        alpha_list_master[frame_number-1][count_edge]
                    )
    
        if len(alpha_list) != 0:
            alpha_average = sum(alpha_list) / len(alpha_list)
            alpha_average_list.append(alpha_average)
            alpha_variance = sum(
                [(val - alpha_average) ** 2 for val in alpha_list]
            ) / (len(alpha_list) - 1)
            alpha_variance_list.append(alpha_variance)
        else:
            alpha_average_list.append(0)
            alpha_variance_list.append(0)
    
        # second half of segments (unchanged logic, with division guard)
        alpha_list = []
    
        for i in range(2, max(nodes_2_dict.keys()), 4):
    
            if len(alpha_list_master[frame_number]) < 22:
    
                if (i, i+1) in edges_2_dict.keys() or (i+1, i) in edges_2_dict.keys():
    
                    r_0 = df_node_pos.loc[i, 1]
                    c_0 = df_node_pos.loc[i, 2]
    
                    r_1 = df_node_pos.loc[i+1, 1]
                    c_1 = df_node_pos.loc[i+1, 2]
    
                    if (r_1 - r_0) != 0:
                        alpha = math.atan((c_1 - c_0) / (r_1 - r_0)) * 180 / np.pi
                    else:
                        alpha = 90.0
    
                    alpha_list.append(alpha)
                    alpha_list_master[frame_number].append(alpha)
    
                    count_edge += 1
    
                    if frame_number > 1:
                        if (alpha_list_master[frame_number-1][count_edge] > 0 and
                                not alpha_list_master_2[count_edge]):
                            alpha_list_master_2[count_edge] = not alpha_list_master_2[count_edge]
                            count += 1
    
                else:
                    count_edge += 1
                    alpha_list_master[frame_number].append(
                        alpha_list_master[frame_number-1][count_edge]
                    )
    
        if len(alpha_list) != 0:
            alpha_average = sum(alpha_list) / len(alpha_list)
            alpha_average_list_2.append(alpha_average)
            alpha_variance = sum(
                [(val - alpha_average) ** 2 for val in alpha_list]
            ) / (len(alpha_list) - 1)
            alpha_variance_list_2.append(alpha_variance)
        else:
            alpha_average_list_2.append(0)
            alpha_variance_list_2.append(0)
    
        alpha_list_vice_master = [
            alpha_list_master[i]
            for i in range(len(alpha_list_master))
            if len(alpha_list_master[i]) > 0
        ]
    
        count_list.append(count)


    
    
    # Function to save the dictionary to an Excel file
    def save_dict_to_excel(data_dict, filename):
        with pd.ExcelWriter(filename) as writer:
            for sheet_name, data in data_dict.items():
                if not sheet_name == -1:
                    if isinstance(sheet_name, int):
                        sheet_name = str(sheet_name)
                    df = pd.DataFrame(data, columns=["frame_number", "pre_row", "pre_col", "new_row", "new_col", "min_distance"])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    
    displacements_output_file_path = f'{test_folder}/Track_data_{test_info}.xlsx'
    # Save the example dictionary to an Excel file
    save_dict_to_excel(node_track_dict_dist, displacements_output_file_path)
    
    
    ## save the graphs to a .pkl file
    # Define the filename
    pkl_filename = f'{test_folder}/graphs_{test_info}.pkl'
    # Save the graph list to a file
    with open(pkl_filename, 'wb') as file:
        pickle.dump(Graphs_List, file)
    
    
    ##################### visualize the thickness and deformation values in pixels
    # and detect the frames where compression starts and ends
    
    # Visualize the output
    plt.figure(figsize=(10, 6))
    plt.plot(thickness_list)        
    plt.title('Overall thickness of the board')
    plt.xlabel('Frame number')
    plt.ylabel('Overall thickness in pixels')
    plt.grid("on")
    
    board_deformation_pixels = thickness_0 - thickness_list
    
    max_value = max(board_deformation_pixels)
    
    min_deformation_index_list = [index for index, element in enumerate(board_deformation_pixels) if element == 0]
    deformation_start_frame = max(min_deformation_index_list)
    
    max_deformation_index_list = [index for index, element in enumerate(board_deformation_pixels) if element == max_value]
    
    deformation_end_frame_1 = min(max_deformation_index_list)
    deformation_end_frame_2 = max(max_deformation_index_list)
    
    # Visualize the output
    plt.figure(figsize=(15,8))
    plt.plot(board_deformation_pixels)
    # Add vertical lines at the highest indices of the minimum and maximum values
    plt.axvline(x=deformation_start_frame, color='r', linestyle='--', label= f'Compression start ({deformation_start_frame})')
    # plt.text(deformation_start_frame, 0, f'Start: {deformation_start_frame}', 
    #                   color='r', ha='center', va='top')
    plt.axvline(x=deformation_end_frame_1, color='g', linestyle='--', label= f'Compression end ({deformation_end_frame_1})')
    plt.axvline(x=deformation_end_frame_2, color='orange', linestyle='--', label=f'Compression end ({deformation_end_frame_2})')
    
    
    # plt.text(deformation_end_frame_1, board_deformation_pixels[deformation_end_frame_1], f'End2: {deformation_end_frame_2}', 
    #                   color='brown', ha='left', va='top')
    
    # plt.title('Estimation of the board deformation in pixels')
    plt.xlabel('Frame number')
    plt.ylabel('Overall deformation in pixels')
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
    ########################################
    # linear approximationof deformation values
    
    board_deformation_pixels_lin = board_deformation_pixels.astype(float) # to work wth nan convert to float first
    
    # Get the indices of the array
    indices = np.arange(len(board_deformation_pixels_lin))
    # Combine indices and values into a new 2D array
    board_deformation_pixels_lin = np.column_stack((indices, board_deformation_pixels_lin))
    
    board_deformation_pixels_lin_filtered = board_deformation_pixels_lin[
        
                    (board_deformation_pixels_lin[:, 0] > deformation_start_frame) & 
                    (board_deformation_pixels_lin[:, 0] <= deformation_end_frame_1)
                    ]
    
    
    # Find unique y values and their first occurrence indices
    _, unique_indices = np.unique(board_deformation_pixels_lin_filtered[:,1], return_index=True)
    
    # Extract rows corresponding to the first occurrence of each unique y value
    board_deformation_pixels_unique_vals = board_deformation_pixels_lin_filtered[unique_indices]
    
    # Separate the x values and f(x) values
    x_values = board_deformation_pixels_unique_vals[:, 0]
    y_values = board_deformation_pixels_unique_vals[:, 1]
    
    # Visualize the output
    plt.figure(figsize=(10, 6))
    plt.scatter( x_values, y_values, color='blue', s=2, label='Unique deformation values')
    plt.show()
    
    # Define the linear function for fitting
    def linear_func(x, a, b):
        return a * x + b
    
    # Perform the linear fit
    params, _ = curve_fit(linear_func, x_values, y_values)
    slope, intercept = params
    
    # Calculate the fitted values for plotting
    fitted_values = linear_func(x_values, slope, intercept)
    
    
    # Solve for f(x) = 0
    x_f0 = -intercept / slope
    # Solve for f(x) = max_deformation + 0.5
    
    max_deformation_approx = board_deformation_pixels_unique_vals[-1,1]+0.5
    x_f_max_deformation = (max_deformation_approx - intercept) / slope
    
    
    x_fit_values = np.linspace(x_f0, x_f_max_deformation, 200)  # Extend slightly beyond the x-intercept
    y_fit_values = linear_func(x_fit_values, slope, intercept)
    
    
    compression_start = round(x_f0)
    compression_end = round(x_f_max_deformation)
    
    if USE_MANUAL_COMPRESSION_FRAMES:
        compression_start = MANUAL_COMPRESSION_START
        if MANUAL_COMPRESSION_END is not None:
            compression_end = MANUAL_COMPRESSION_END
    
    
    plt.figure(figsize=(15, 8))
    
    plt.plot(board_deformation_pixels)
    # Plot the fitted linear line
    plt.plot(x_fit_values, y_fit_values, color='orange', linestyle='--', linewidth = 2, label='Linear Fit')
    
    # Plot the original data points
    plt.scatter(x_values, y_values, color='black' , s=10 , label='Unique values')
    
    plt.axvline(x=compression_start, color='r', linestyle='--', label= f'Compression start ({compression_start})')
    
    plt.axvline(x=compression_end, color='g', linestyle='--', label= f'Compression end ({compression_end})')
    plt.axhline(y=max_deformation_approx, color='brown', linestyle='-', label= 'Maximum deformation')

    # Add labels and title
    plt.xlabel('Frame number')
    plt.ylabel('Defromation in pixels')
    # plt.title('Linear approximation of the deformation in pixels as a function of frame number')
    plt.legend(fontsize=20)
    # plt.legend()

    plt.grid("on")
    
    # Show the plot
    plt.show()
    
    
    
    
    
    
    
    # threshold_number_inv_angles = round(max(count_list)//10 +1)
    
    threshold_buckling_end = round(max(count_list)*0.8)

    threshold_number_inv_angles = 1
    
    frame_pos_peak_D = next((index for index, value in enumerate(count_list) 
                                   if value >= threshold_number_inv_angles), None)
    
    frame_buckling_end = next((index for index, value in enumerate(count_list) 
                                   if value >= threshold_buckling_end), None)
    
    # Plotting the data
    plt.figure(figsize=(15,10))
    plt.plot(count_list, )
    plt.axvline(x = frame_pos_peak_D, linestyle='--', color ='r', label=f'Buckling start at {frame_pos_peak_D}', linewidth = 3)
    plt.axvline(x = frame_buckling_end, linestyle='--', color ='g', label=f'Buckling end at {frame_buckling_end}', linewidth = 3)
    
    # plt.axhline(y = threshold_number_inv_angles, linestyle='--', label='Threshold', linewidth = 1)

    plt.xlabel('Frame number')
    plt.ylabel('Number of angles with inverted sign')
    plt.title('Limits of the buckling region')
    # Setting the y-ticks to steps of 5 degrees
    # plt.yticks(np.arange(0, 23, 2))
    # plt.ylim(0,16)
    
    # Setting the y-ticks to steps of 5 degrees
    # plt.xlim(220,260)
    
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    
    
    # Plotting the data
    plt.figure(figsize=(14,14))
    plt.plot(alpha_list_vice_master)
    
    # Setting the y-ticks to steps of 5 degrees
    plt.yticks(np.arange(-45, 45, 5))
    
    plt.xlabel('Frame number')
    plt.ylabel('Angle in degrees')
    plt.title('Angle between fluting segements and the vertical')
    
    plt.ylim(-45,45 )
    plt.xlim(0, compression_end)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    
    # Plotting the data
    plt.figure(figsize=(17,17))
    
    plt.plot(alpha_average_list_2, label='Angle of ascending wave segments')
    plt.plot([mu + sigma for mu, sigma in zip(alpha_average_list_2, np.sqrt(alpha_variance_list_2))], label='Variance Upper Limit of Angle of ascending wave segments')
    plt.plot([mu - sigma for mu, sigma in zip(alpha_average_list_2, np.sqrt(alpha_variance_list_2))], label='Variance Lower Limit of Angle of ascending wave segments')
    
    plt.plot(alpha_average_list, label='Angle of descending wave segments')
    plt.plot([mu + sigma for mu, sigma in zip(alpha_average_list, np.sqrt(alpha_variance_list))], label='Variance Upper Limit of Angle of descending wave segments')
    plt.plot([mu - sigma for mu, sigma in zip(alpha_average_list, np.sqrt(alpha_variance_list))], label='Variance Lower Limit of Angle of descending wave segments')
    
    
    # Setting the y-ticks to steps of 5 degrees
    plt.yticks(np.arange(-45, 45, 5))
    
    plt.xlabel('Frame number')
    plt.ylabel('Angle in degrees')
    plt.title('Angle between fluting segements and the vertical')
    
    plt.ylim(-45,45 )
    plt.xlim(0, compression_end)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    image_analysis_output_data = []
    
    thickness_min_pixel_img = min(thickness_list)

    delta_H_pixel_image = max(board_deformation_pixels)
    image_analysis_output_data.append((compression_start,
                                       compression_end,
                                       frame_pos_peak_D,
                                       frame_buckling_end,
                                       delta_H_pixel_image,
                                       thickness_0,
                                       thickness_min_pixel_img))
    
    thickness_0_pixels_img = thickness_0
    # Create a DataFrame from the collected data
    image_analysis_output_df = pd.DataFrame(image_analysis_output_data, columns=[ 'compression_start',
                                                                   'compression_end',
                                                                   'frame_pos_peak_D',
                                                                   'frame_buckling_end',
                                                                   'delta_H_pixel_image',
                                                                   'thickness_0_pixels_img',
                                                                   'thickness_min_pixel_img'] )
                    
    # Save the DataFrame to a new Excel file
    image_analysis_output_df.to_excel(image_analysis_output_file_path, sheet_name= 'img_analy_data' , index=False)
    


##############################################################################
# Create a figure with 4 subplots to plot 4 sellected frames
fig_master, axs_master = plt.subplots(4, 1, figsize=(10, 10))
frames_to_plot = [1, 50, 90, 125]
titles_list = ['Initial state', 'Early compression', 'Middle of compression', 'Late compression']
# Plot images on the subplots
for i in range(len(frames_to_plot)):
    # Simulate 4 images with random data for demonstration
    current_img_path = os.path.join(destination_folder_graphs, f"Graph_Frame_{frames_to_plot[i]:04d}.jpg")
    current_gray_img = cv2.imread(current_img_path)
    current_gray_img = cv2.cvtColor(current_gray_img, cv2.COLOR_BGR2RGB)
    axs_master[i].imshow(current_gray_img)
    axs_master[i].axis('off')  # Hide the x and y axes
    axs_master[i].set_title(titles_list[i],  fontsize = 20 )
    axs_master[i].tick_params(axis='both', which='major', labelsize=8)  # Set tick label size

# Adjust the layout to prevent overlap
# plt.tight_layout()
# Display the figure with images
plt.show()
##############################################################################


    
#%% Read experimental data and synchronize it with the sample video, detect frames of load peaks

# Functions to read the graphs data in pixels and in mm

# define a figure size 
fig_size_plots = 9

#############################################################################

def graph_data (graph):

    # read the nodes
    nodes = graph.nodes()
    
    # sorted list of the nodes
    nodes_list = sorted(list(nodes))
    
    # extract the nodes to be used in the model
    nodes_list = nodes_list[ nodes_limits[0]: nodes_limits[1]+1]
    
    # initiate an array of zeros with number of rows equal to the number of nodes
    # and with 3 columns for [node_name , node_row_position , nodes_column_position]
    nodes_coord_array = np.zeros( (len(nodes_list),3) ).astype(int)
    
    # initiate a counter of the node number to 0
    counter = 0
    
    # for loop to read and store the nodes data in the array as follows:
    # every row containes: [node_label , node_row_position , node_column_position]
    for node in nodes:
        
        # get the data on the nodes chosen only
        if node in nodes_list:
            
            # read node data
            node_data_list = [node, nodes[node]['o'][0] , nodes[node]['o'][1]]
            # add the node data to the array
            nodes_coord_array[counter,:] = np.array(node_data_list)
            # counter incrementation
            counter +=1
      
    # Sort the array based on the nodes labels
    nodes_coord_array = nodes_coord_array[nodes_coord_array[:, 0].argsort()]
    
    # read the maximum raw value
    max_row_val = nodes_coord_array[:,1].max()
    
    # calculate and save the coordinates of nodes in the new system of coordinates 
    # for the model, which is not the same as the system of coordinates of an image
    # initiate a new array with the same shape, to store the new node data
    nodes_coord_array_new_ref = np.zeros(nodes_coord_array.shape)
    
    # initiate a dict for the node positions in the new 
    node_pos_dict = {}
    
    for index , (node, row, col) in enumerate(nodes_coord_array):
        
        # calculate the x-value
        new_x = col
        # calculate the y-value
        new_y = max_row_val - row
        # add the coordinates to the array
        nodes_coord_array_new_ref[index, : ] = [node, new_x, new_y]
        # convert the array to int
        nodes_coord_array_new_ref = nodes_coord_array_new_ref.astype(int)
        # add the node coordinates to the dict with node lable as key
        node_pos_dict[node] = (new_x,new_y)
    
    # calculate a mean value for the y-positions
    mean_y = nodes_coord_array_new_ref[:,2].mean()
        
    # nodes with y-positions over the mean value are upper liner nodes
    nodes_coord_array_upper_liner = nodes_coord_array_new_ref[nodes_coord_array_new_ref[:,2] > mean_y].astype(int)
    # nodes with y-positions over the mean value are lower liner nodes
    nodes_coord_array_lower_liner = nodes_coord_array_new_ref[nodes_coord_array_new_ref[:,2] <= mean_y].astype(int)
    
    # calculate an average value for the upper and lower liner vertical positons
    upper_liner_avr = np.mean(nodes_coord_array_upper_liner[:,2])
    lower_liner_avr = np.mean(nodes_coord_array_lower_liner[:,2])
    # estimate a thickness of the fluting.
    thickness_estimation = round(upper_liner_avr- lower_liner_avr)

    # initiate and save the upper and lower liner nodes in dicts
    node_pos_dict_upper_liner = {}
    node_pos_dict_lower_liner = {}
    
    for node, x, y in nodes_coord_array_upper_liner:
        node_pos_dict_upper_liner[node] = (x,y)
    
    for node, x, y in nodes_coord_array_lower_liner:
        node_pos_dict_lower_liner[node] = (x,y)
    

    
    return (node_pos_dict, 
            node_pos_dict_upper_liner, 
            node_pos_dict_lower_liner,
            nodes_list,
            thickness_estimation)


def graph_data_mm(graph, pixel_mm ):

    # read the nodes
    nodes = graph.nodes()
    
    # sorted nodes list
    nodes_list = sorted(list(nodes))
    # extract only the nodes chosen for the modeling
    nodes_list = nodes_list[ nodes_limits[0]: nodes_limits[1]+1]
    
    # initiate an array of zeros with number of rows equal to the number of nodes
    # and with 3 columns for [node_name , node_row_position , nodes_column_position]
    nodes_coord_array = np.zeros( (len(nodes_list),3) ).astype(int)
    
    # initiate a counter of the node number to 0
    counter = 0
    
    # for loop to read and store the node information in the array as follows:
    # every row containes: [node_name , node_row_position , nodes_column_position]
    for node in nodes:
        
        if node in nodes_list:
            
            # read the node data
            node_data_list = [node, nodes[node]['o'][0] , nodes[node]['o'][1]]
            # add it to the array
            nodes_coord_array[counter,:] = np.array(node_data_list)
            # increment the counter
            counter +=1
            
        
    # Sort the array based on nodes labels
    nodes_coord_array = nodes_coord_array[nodes_coord_array[:, 0].argsort()]
    # read the maximum raw value
    max_row_val = nodes_coord_array[:,1].max()
    
    # calculate and save the coordinates of nodes in the new system of coordinates 
    # for the model, which is not the same as the system of coordinates of an image
    # initiate a new array with the same shape, to store the new node data
    nodes_coord_array_new_ref = np.zeros(nodes_coord_array.shape)
    
    
    for index , (node, row, col) in enumerate(nodes_coord_array):
        # calculate the new x value
        new_x = col
        # calculate the new y value
        new_y = max_row_val - row
        # add the new coordinates to the array
        nodes_coord_array_new_ref[index, : ] = [node, new_x, new_y]
        
    
    # convert the array to int
    nodes_coord_array_new_ref = nodes_coord_array_new_ref.astype(int)

    # calculate a mean value of the nodes y-positions
    mean_y = nodes_coord_array_new_ref[:,2].mean()
    
    # nodes with y-positions over the average are upper liner nodes
    nodes_coord_array_upper_liner = nodes_coord_array_new_ref[nodes_coord_array_new_ref[:,2] > mean_y]
    # nodes with y-positions over the average are lower liner nodes
    nodes_coord_array_lower_liner = nodes_coord_array_new_ref[nodes_coord_array_new_ref[:,2] < mean_y]
    
    # calculate an average y-position for upper and lower liner nodes
    upper_liner_avr = np.mean(nodes_coord_array_upper_liner[:,2])
    lower_liner_avr = np.mean(nodes_coord_array_lower_liner[:,2])
    # estimate a fluting thickness
    thickness_estimation = (upper_liner_avr- lower_liner_avr)*pixel_mm

    # initiate dict for the node coordinates
    node_pos_dict = {}
    node_pos_dict_upper_liner = {}
    node_pos_dict_lower_liner = {}
    
    # calculate the distances in mm for each node position and add the values to
    # dicts with node label as key
    for node, x, y in nodes_coord_array_new_ref:
        x_mm = round(x*pixel_mm,3)
        y_mm = round(y*pixel_mm,3)
        node_pos_dict[node] = (x_mm ,y_mm)
    
    # do the same for the upper liner nodes only
    for node, x, y in nodes_coord_array_upper_liner:
        x_mm = round(x*pixel_mm,3)
        y_mm = round(y*pixel_mm,3)
        node_pos_dict_upper_liner[node] = (x_mm,y_mm)
    
    # do the same for the lower liner nodes only
    for node, x, y in nodes_coord_array_lower_liner:
        x_mm = round(x*pixel_mm,3)
        y_mm = round(y*pixel_mm,3)
        node_pos_dict_lower_liner[node] = (x_mm,y_mm)
    

    
    return (node_pos_dict, 
            node_pos_dict_upper_liner, 
            node_pos_dict_lower_liner,
            nodes_list,
            thickness_estimation)


##############################################################################
# Thickness and maximum deformation estimation of the board between the 
# initial and final states

# File name containing the graphs of a given sample
graph_data_filename = f'{test_folder}/graphs_{test_info}.pkl'

# Load the graph list from the file
with open(graph_data_filename, 'rb') as file:
    Loaded_Graphs_List = pickle.load(file)

print("Graphs have been loaded.")

# save the first graph to a variable 
graph_0 = Loaded_Graphs_List[compression_start]

# call function to extract data from the first graph
(node_pos_dict_0,
node_pos_dict_upper_liner_0,
node_pos_dict_lower_liner_0,
nodes_list_0,
thickness_estimation_0                  ) = graph_data(graph_0)

# save the graph corresponding to the end of compression to a variable 
graph_n = Loaded_Graphs_List[ compression_end ]

# call function to extract data from the last graph
(node_pos_dict_n, 
node_pos_dict_upper_liner_n, 
node_pos_dict_lower_liner_n,
nodes_list_n,
thickness_estimation_n                  ) = graph_data(graph_n)

# calculate a deformation estimation in pixels between the first and last frames
delta_H_pixel = thickness_estimation_0 - thickness_estimation_n


##############################################################################
# Processing of the measurement data from the excel file of the FCT


# rename the data frame columns
df.columns = ['Crush (mm)', 'Force (N)']

# Convert values of the data frame to numeric, none numeric values will be set to NaN
df['Force (N)'] = pd.to_numeric(df['Force (N)'], errors='coerce')
df['Crush (mm)'] = pd.to_numeric(df['Crush (mm)'], errors='coerce')

# Drop rows with NaN values
df = df.dropna(subset=['Force (N)']).reset_index(drop=True)

# Remove initial rows with negative crush values
first_non_negative_index = df[df['Crush (mm)'] > 0].index[0]
df = df.iloc[first_non_negative_index:].reset_index(drop=True)

# Round the 'Displacement_New' to 3 decimal places and 'Force (N)_New' to 2 decimal places
df['Crush (mm)'] = df['Crush (mm)']
df['Force (N)'] = df['Force (N)']

# reset first crush value to zero
# df.loc[0,'Crush (mm)'] = 0

# Remove the last 23 measurements because there is practically no displacement
# happening. The upper plate is not moving, there is no compression and the load
# is dropping drastically.
df = df.iloc[:-25]

# Extract the peak indices based on the raw measurement data
# Convert the df Force column to nd array
array_Force = df['Force (N)'].to_numpy()
# Find index of peaks, local maximas with order = 5 meaning that the value of 
# the local maxima has to higher than 5 values to the left and 5 to the right.
peak_indices_1 = argrelmax(array_Force, order = round (0.07*array_Force.size))[0]
# Extract the rows corresponding to the peaks into a new DataFrame
peak_df_1 = df.iloc[peak_indices_1].reset_index(drop=True)


# #############################################################################
# The zwick in this FCT test follows a prescribed displacement (comstant movement 
# speed of the upper plate). Crush(mm) is linear in the time and the measurement 
# rate is not constant. That is when the Load gradiant is higher, the measurements 
# are more frequent, while the deformation speed is constant. 
# The non linearity between crush values and measurement points can be seen in the next figures

# So we can not just add a frames column to the measurements data frame, as linearity
# is not given. Inseat a coefficient between every Crush value in mm and the 
# corresponding Frame number need to be calculated.


# Plotting the Crush(mm) as a function of the mesurement points
plt.figure(figsize=(fig_size_plots,fig_size_plots))
plt.plot(df['Crush (mm)'])

plt.xlabel('Measurement points')
plt.ylabel('Crush (mm)')
plt.title('Displacement measurements')

plt.tight_layout()
plt.grid(True)
# plt.show()
plt.close()

# Plotting the mesurement points as a function of the displacement
plt.figure(figsize=(fig_size_plots,fig_size_plots))
plt.plot(df['Crush (mm)'], range(len(df['Crush (mm)'])))

plt.xlabel('Crush (mm)')
plt.ylabel('Measurement points')
plt.title('Displacement measurements')

plt.tight_layout()
plt.grid(True)
# plt.show()
plt.close()

##############################################################################
############ first wrong method to add a frames column linearly
# Determine the length of the DataFrame columns
# length_of_df = len(df)
# compression_frames_List = list(range(compression_start, compression_end))
# # Create a Frame_n from the example list that matches the length of the DataFrame columns
# frames_Frame_n = np.linspace(compression_frames_List[0], compression_frames_List[-1], length_of_df).round().astype(int)
# # Add the Frame_n list as a new column to the DataFrame
# df['Frame_n'] = frames_Frame_n
##############################################################################


# calculation of the coefficient coeff of a function that has the form: 
#   frame_n  =  coeff * Crush (mm) + compression_start

# The inverce function is: 
#   Crush (mm) = (Frame_n - compression_start) / coeff

# For the last frame commpression_end havin the maximum last Crush (mm) value 
#   commpression_end  =  coeff * Crush_last_value + compression_start

# calculate the coefficient
coeff = (compression_end - compression_start) / df['Crush (mm)'].iloc[-1]

# calculate the frame number for every crush value and round it to an integer type
df['Frame_n'] = (round(coeff * df['Crush (mm)'] + compression_start)).astype(int)


# # Plotting the Crush (mm) as a function of frame number ( it is now linear )  
plt.figure(figsize=(fig_size_plots,fig_size_plots))
plt.plot(df['Frame_n'], df['Crush (mm)'])

plt.xlabel('Measurement points',  fontsize = 20)
plt.ylabel('Crush (mm)',  fontsize = 20)
plt.title('Crush (mm) measurements',  fontsize = 20)

plt.tight_layout()
plt.grid(True)
# plt.show()
plt.close()


# Calculate the pixel resolution based on the deformation estimation in pixels
# and the deformation measured in mm
delta_H_mm = df['Crush (mm)'].iloc[-1] - df['Crush (mm)'].iloc[0]
pixel_mm = round(delta_H_mm / delta_H_pixel, 6)

pixel_mm_image_based = round(delta_H_mm / delta_H_pixel_image, 6) 

# pixel_mm = 0.087
# pixel_mm_image_based = 0.087

# calculate the initial thickness estimation in mm based on graphs thickness values
thickness_estimation_0_mm_graph = thickness_estimation_0 * pixel_mm

# calculate the initial thickness estimation in mm based on image analysis thickness values
thickness_estimation_0_mm = thickness_0_pixels_img * pixel_mm_image_based

# calculate the deformation values in percent with reference to the initial thickness in mm
df['Strain'] = (df['Crush (mm)']- df['Crush (mm)'].iloc[0]) / thickness_estimation_0_mm
df['Stress (MPa)'] = df['Force (N)']/ sample_area


array_stress = df['Stress (MPa)'].to_numpy()
array_strain = df['Strain'].to_numpy()
# Step 1: Compute the derivative of the signal
array_stress_derivative = np.gradient(array_stress,array_strain)

# array_stress_derivative = gaussian_filter1d(array_stress_derivative, sigma= len(array_stress_derivative) //400 )  # Replace `stress` with your data
derivative_peaks = argrelmax(array_stress_derivative, order = round (0.05*array_stress_derivative.size) )[0]
derivative_peak_1_pos = derivative_peaks[0]
derivative_peak_1_value = array_stress_derivative[derivative_peak_1_pos]


# Step 4: Find indices on the left and right that are 10% different from the peak value
stress_threshold = 0.3 * derivative_peak_1_value

# Find the index on the left
left_index = derivative_peak_1_pos - 1
while left_index > 0 and abs(array_stress_derivative[left_index] - derivative_peak_1_value) < stress_threshold:
    left_index -= 1

frame_elasticity_start = df.loc[left_index,'Frame_n']

# Find the index on the right
right_index = derivative_peak_1_pos + 1
while right_index < len(array_stress_derivative) and abs(array_stress_derivative[right_index] - derivative_peak_1_value) < stress_threshold:
    right_index += 1

frame_elasticity_end = df.loc[right_index,'Frame_n']

plt.figure(figsize=(20, 10))
# Plotting the derivative of the stress with respect to strain
plt.plot(array_strain, array_stress_derivative)
plt.plot(array_strain[derivative_peak_1_pos], array_stress_derivative[derivative_peak_1_pos], 'ro', label='First peak')


plt.axhline(y = 0.7 * derivative_peak_1_value, color='r', linestyle='--', linewidth=1, label='70% of peak value')  # Draw vertical line


plt.axvline(array_strain[left_index], color='g', linestyle='--', label='Modeling interval')
plt.axvline(array_strain[right_index], color='g', linestyle='--')
plt.title('Finite differences of stresses')
plt.xlabel('Strain')
plt.ylabel('Δσ in MPa')
plt.legend()
plt.grid("on")
plt.tight_layout()
plt.show()

x1 = df.loc[left_index,'Strain']
y1 = df.loc[left_index,'Stress (MPa)']


x2 = df.loc[right_index,'Strain']
y2 = df.loc[right_index,'Stress (MPa)']


# Plotting the original signal
plt.figure(figsize=(20, 10))
plt.plot(array_strain, array_stress, linewidth = 2 )

plt.plot([x1, x2], [y1 , y2], color = 'red', label = 'linear approximation', linewidth = 2)

plt.title('Stress-strain curve')
plt.xlabel('Strain')
plt.ylabel('σ in MPa')
plt.axvline(array_strain[left_index], color='g', linestyle='--', label='Quasi-linear region')
plt.axvline(array_strain[right_index], color='g', linestyle='--')
plt.legend()
plt.grid("on")
plt.show()



# calculate the time in percent based on the frame numbers, which are linear in time.
# and add it to as a column to the data frame
df['Rel_Time (%)'] = round( 100*(df['Frame_n']- compression_start) / (compression_end-compression_start), 2)


##############################################################################
# As Frames are unique, the column frame number in the data frame must to contain unique values
# A data frame df_unique is created where frame numbers are unique

# Find the last index of a given Frame_n value
def find_last_index(df, column_name, value):
    
    # Find all indices of the given value
    indices = df[df[column_name] == value].index
    if len(indices) > 0:
        # Return the last index
        return indices[0]
    else:
        print('The Index varibles are empty')
        return None


# Extract all the lines with the last index for each Frame number value
unique_values = df['Frame_n'].unique()
indices_to_extract = [find_last_index(df, 'Frame_n', value) for value in unique_values]

# Filter the DataFrame to include only the rows with these indices to extract
df_unique = df.iloc[indices_to_extract].reset_index(drop=True)






# Add columns to the data frame indicating the relative and absolute time in seconds:
# relative timer starts when the compression starts
# absolute timer starts from the beginning of the video, including the pre-compression phase
df_unique['Rel_Time (s)'] = round(df_unique['Frame_n'] / new_fps - df_unique['Frame_n'].iloc[0] / new_fps, 2)
df_unique['Abs_Time (s)'] = round(df_unique['Frame_n'] / new_fps, 2)


# Convert the df Force column to nd array
array_Force_2 = df_unique['Force (N)'].to_numpy()


# Find index of peaks, local maximas with order = 5 meaning that the value of 
# the local maxima has to higher than 5 values to the left and 5 to the right.
peak_indices_2 = argrelmax(array_Force_2, order = round (0.07*array_Force_2.size) )[0]

# Extract the rows corresponding to the peaks into a new DataFrame
peak_df = df_unique.iloc[peak_indices_2]

# concat the first and last compression frames with their data.
peak_df = pd.concat( [df_unique.iloc[[0]], 
                                peak_df,
                                df_unique.iloc[[-1]]
                                ], axis=0)

# reset the index values to start from 0
peak_df = peak_df.reset_index(drop=True)



# Change display option to show all columns of the data frame in the console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)        # change display width
pd.set_option('display.colheader_justify', 'left')  # Justify headers Justified the left

# # Reset the specific display options to default
# pd.reset_option('display.max_columns')
# pd.reset_option('display.width')
# pd.reset_option('display.colheader_justify')
# # or 
# # pd.reset_option('all')

# print the peaks data
print(peak_df)



# plotting with 2 x axis: deformation in mm and the corresponding Frames number 
fig, ax1 = plt.subplots(figsize=(24, 10))

# Plot with the first x-axis (crush in mm)
ax1.plot(df_unique['Crush (mm)'], df_unique['Force (N)'], color='b')
ax1.set_xlabel('Deformation in mm')
ax1.set_ylabel('Load (N)')
ax1.set_xlim(0,3)
ax1.set_ylim(0,800)
ax1.set_title('Load-Deformation curve')

# Create the frames axis
ax2 = ax1.twiny()

ax2.set_xlabel('Frame number')
# Synchronize the 2 axis
ax2.set_xlim(ax1.get_xlim())

# Set  ticks and labels for the second x-axis
ax2.set_xticks(df_unique['Crush (mm)'][::(50//fps_div)])
ax2.set_xticklabels(df_unique['Frame_n'][::(50//fps_div)])



# plot vertical lines indicating the position of peaks on the first axis
for ii in range(1, peak_df.shape[0]-1):
    
    # variable to lable the peaks with letters A, B, C and D
    letter = chr(ord('A')+ii-1)
    # Draw the vertical line at the mapped value
    
    
    peak_i_pos_crush = peak_df['Crush (mm)'][ii]
    peak_i_pos_Frame = peak_df['Frame_n'][ii]
    ax1.axvline(x= peak_i_pos_crush, color=f'C{ii}', linestyle='--', label=f'Peak {letter} at {peak_i_pos_Frame}')

    
# # Draw a vertical line indicating the end of compression.
ax1.axvline(x=df_unique['Crush (mm)'].iloc[-1], color='brown', linestyle='--', label='End')


# Combine legends from both axes

ax1.legend()

# Adding more ticks on the displacement axis in mm
x_ticks = np.arange(0, 3, 0.1) 
ax1.set_xticks(x_ticks)
ax1.grid("on")
plt.show()
plt.close()


    
peaks_excel_filename = f'{test_folder}/peaks_data_{meth}_{sample_name}_rate{fps_div}.xlsx'
# Save the DataFrame to an Excel file
peak_df.to_excel(peaks_excel_filename, index=True)





#%% load peaks prediction and nodes clustering based on the displacement computation


if Node_clustering : 
    
    # Load the Excel file
    displacements_file_path = f'{test_folder}/Track_data_{test_info}.xlsx'  # Update the path if needed
    
    excel_data = pd.read_excel(displacements_file_path, sheet_name=None)
    
    # Function to calculate the distance
    def calculate_distance(pre_row, pre_col, new_row, new_col):
        return ((new_row - pre_row) ** 2 + (new_col - pre_col) ** 2) ** 0.5
    
    
    # Function to calculate the distance
    def calculate_distancex(pre_row, pre_col, new_row, new_col):
        return new_col - pre_col
    
    # Function to calculate the distance
    def calculate_distancey(pre_row, pre_col, new_row, new_col):
        return new_row - pre_row
    
    # Function to calculate average for each position across lists
    def node_groups_average(distances_list):
        
        groups_disp_lists = []
        
        for group in range(4):
            groups_disp_lists.append([distances_list[i] for i in range(group, len(distances_list) , 4)])
        
        groups_avr_lists = []
    
        for group in range(4):
            
            average_list = []
            for values in zip_longest(*groups_disp_lists[group], fillvalue=None):
                # Filter out None values and calculate the average for current position
                valid_values = [v for v in values if v is not None]
                avg = sum(valid_values) / len(valid_values)
                average_list.append(avg)
                
            
            average_list[:compression_start-1] = [0] * compression_start
            
            # average_list = average_list[:compression_end]
            groups_avr_lists.append(average_list)
        
        return groups_avr_lists
    
    
    
    track_file = os.path.join(test_folder, f"Track_data_{test_info}.xlsx")
    excel_data = pd.read_excel(track_file, sheet_name=None)

        # define a reference for distance calculation
    Ref_frame = compression_start

    # ------------------------------------------------------------------
    # SAVE per-node displacements (relative to FIRST frame of each node)
    # ------------------------------------------------------------------
    disp_rows = []

    for sheet_name, data in excel_data.items():
        if data.empty:
            continue

        # each sheet corresponds to one node
        try:
            node_id = int(sheet_name)
        except ValueError:
            # skip non-numeric sheet names if any
            continue

        # reference position = first row for this node (its first frame)
        ref_row = data.iloc[0]['pre_row']
        ref_col = data.iloc[0]['pre_col']

        # for every row (frame) of this node, compute dx, dy, dist
        for _, row in data.iterrows():
            frame = int(row['frame_number'])
            new_row = row['new_row']
            new_col = row['new_col']

            dx = new_col - ref_col        # x displacement in pixels
            dy = new_row - ref_row        # y displacement in pixels
            dist = math.hypot(dx, dy)     # Euclidean displacement

            disp_rows.append({
                "node_id":      node_id,
                "frame_number": frame,
                "dx_px":        float(dx),
                "dy_px":        float(dy),
                "dist_px":      float(dist),
            })

    if disp_rows:
        disp_df = pd.DataFrame(disp_rows)
        disp_csv_path = os.path.join(
            test_folder,
            f"node_displacements_firstFrameRef_{test_info}.csv",
        )
        disp_df.to_csv(disp_csv_path, index=False)
        print(f"Saved per-node displacements to: {disp_csv_path}")
    else:
        print("No displacement data found to save (disp_rows is empty).")

    # choose 2 frames to extract positions of nodes in them and do the clustering
    clustering_Frame_1 = compression_end - 1
    clustering_Frame_2 = round((compression_start + compression_end) / 2)

    
    # helper: get the pre_row/pre_col at the reference frame for this node
    def get_reference_pre_coords(data, Ref_frame):
        """
        data: DataFrame for one node (one Excel sheet)
        Ref_frame: global frame number (e.g. compression_start)
        returns (pre_row, pre_col) or (None, None) if that frame is not present
        """
        ref_rows = data[data["frame_number"] == Ref_frame]
        if ref_rows.empty:
            return None, None
        ref = ref_rows.iloc[0]
        return ref["pre_row"], ref["pre_col"]
    
    # choose 2 frames to extract positions of nodes in them and do the clustering
    clustering_Frame_1 = compression_end -1
    clustering_Frame_2 = round((compression_end-compression_start)*3/4 + compression_start)
    Frame_2 = compression_end + round((compression_end-compression_start)/10)
    
    threshold_disp_curves = 1.5
    alpha_transparency = 0.3
    
    
    
    ##############################################################################
    ################################ Absolute displacements 
    
    plt.figure(figsize=(25, 15))
    # calculate the absolute distances
    for sheet_name, data in excel_data.items():
        if not data.empty:
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
            if pre_row is None:
                # no data for this node at the reference frame
                continue

            distances = [calculate_distance(pre_row, pre_col, new_row, new_col)
                         for new_row, new_col in zip(data['new_row'], data['new_col'])]
            plt.plot(distances)
            
    plt.axvline(x=compression_start, color='brown', linestyle='--', label= f'Compression start {compression_start}', linewidth = 4)

    plt.axvline(x=clustering_Frame_1, color='black', linestyle='--', label= f'Clustering at frame {clustering_Frame_1}', linewidth = 4)
    # plt.axvline(x=compression_end, color='brown', linestyle='--', label= f'Compression end {compression_end}', linewidth = 4)

    plt.xlabel('Frame number')
    plt.ylabel('Displacement in pixels')
    plt.title('Absolute displacement of nodes over the frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    
    distances_list = []
    # Plotting the distances
    plt.figure(figsize=(20, 15))

    # calculate the absolute distances
    for sheet_name, data in excel_data.items():
        if not data.empty:
            pre_row, pre_col = get_reference_pre_coords(data, Ref_frame)
            if pre_row is None:
                continue

            distances = [calculate_distance(pre_row, pre_col, new_row, new_col)
                         for new_row, new_col in zip(data['new_row'], data['new_col'])]

            color_number = int(sheet_name) % 4
            distances_list.append(distances)

            plt.plot(distances, alpha=alpha_transparency, color=f'C{color_number}')
    
    groups_avr = node_groups_average(distances_list)
    for group in range(len(groups_avr)):
        plt.plot(groups_avr[group], label = f"group {group+1}", linewidth = 3)
    
    
    # plot vertical lines indicating the position of peaks on the first axis
    for ii in range(1, peak_df.shape[0]-1):
        
        color_i = 3+ii
        # variable to lable the peaks with letters A, B, C and D
        letter = chr(ord('A')+ii-1)
        
        # Draw the vertical line at the mapped value
        peak_i_pos_Frame = peak_df['Frame_n'][ii]
        
        plt.axvline(x=peak_i_pos_Frame, color= f'C{color_i}', linestyle='--', label=f'Peak {letter} at {peak_i_pos_Frame}', linewidth = 3)
        

    
    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x= compression_start , color='black', linestyle='--', label=f'Compression start {compression_start}', linewidth = 3)

    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x= compression_end, color='black', linestyle='--', label= f'Compression end {compression_end}', linewidth = 3)
    
    # Adding more ticks on the displacement axis in mm
    # x_ticks = np.arange(0, 215, 10) 
    # plt.xticks(x_ticks)

    plt.xlabel('Frame No.')
    plt.ylabel('Displacement (px)')
    plt.title('Displacement Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    
    
    
    frames_pos_list_abs = []
    for list_i in groups_avr :
        frames_pos_list_abs.append (next((index for index, value in enumerate(list_i) 
                                           if value > threshold_disp_curves), None))
    
    # Plotting the distances
    plt.figure(figsize=(20, 15))
    text_pos = 16
    
    
    
    for group in range(len(groups_avr)):
        
        plt.plot(groups_avr[group], label = f"group {group+1}")
        
        plt.axvline(x=frames_pos_list_abs[group], color='r', linestyle='--', linewidth=1)  # Draw vertical line
        plt.text(frames_pos_list_abs[group], text_pos+group, f"{frames_pos_list_abs[group]}", verticalalignment='bottom', color='r')  # Add text annotation
        
    plt.axhline(y = threshold_disp_curves, color='r', linestyle='--', linewidth=1)  # Draw vertical line
    
    plt.xlabel('Frame No.')
    plt.ylabel('Displacement (px)')
    plt.title('Displacement Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    ##############################################################################
    ################################ y displacements 
    
    
    
    
    plt.figure(figsize=(25, 15))
    # calculate the absolute distances
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
            distances = [calculate_distancey(pre_row, pre_col, new_row, new_col) for new_row, new_col in zip(data['new_row'], data['new_col'])]
            plt.plot(distances)   #  label=str(sheet_name)

    
    plt.axvline(x=compression_start, color='brown', linestyle='--', label= f'Compression start {compression_start}', linewidth = 4)
    plt.axvline(x=compression_end, color='black', linestyle='--', label= f'Compression end {compression_end}', linewidth = 4)
    # plt.axvline(x=compression_end, color='brown', linestyle='--', label= f'Compression end {compression_end}', linewidth = 4)

    plt.xlabel('Frame No.')
    plt.ylabel('Displacement (x)')
    plt.title('Vertical displacement of nodes over the frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    distancesy_list = []
    
    # Plotting the y distances
    plt.figure(figsize=(20, 15))
    
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
            distancesy = [calculate_distancey(pre_row, pre_col, new_row, new_col) for new_row, new_col in zip(data['new_row'], data['new_col'])]
            
            # distancesy = distancesy[:compression_end]
    
            distancesy_list.append(distancesy)
            
            color_number = int(sheet_name) % 4    
            plt.plot(distancesy, alpha=alpha_transparency , color = f'C{color_number}')   #  label=str(sheet_name)
    
    
    groups_avr_y = node_groups_average(distancesy_list)
    
    for group in range(len(groups_avr_y)):
        plt.plot(groups_avr_y[group], label = f"group {group+1}", linewidth = 3)
    
    # plot vertical lines indicating the position of peaks on the first axis
    for ii in range(1, peak_df.shape[0]-1):
        
        color_i = 3+ii
        # variable to lable the peaks with letters A, B, C and D
        letter = chr(ord('A')+ii-1)
        # Draw the vertical line at the mapped value
        
        peak_i_pos_Frame = peak_df['Frame_n'][ii]
        
        plt.axvline(x=peak_i_pos_Frame, color= f'C{color_i}', linestyle='--', label=f'Peak {letter} at {peak_i_pos_Frame}', linewidth = 3)
        
    
    # x = df_unique['Frame_n'].iloc[0]              x = df_unique['Frame_n'].iloc[-1]
    
    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x= compression_start , color='black', linestyle='--', label= f'Compression start {compression_start}', linewidth = 3)

    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x=compression_end, color='black', linestyle='--', label= f'Compression end {compression_end}', linewidth = 3)
    
    # Adding more ticks on the displacement axis in mm
    # x_ticks = np.arange(0, 215, 10) 
    # plt.xticks(x_ticks)
    

    plt.xlabel('Frame No.')
    plt.ylabel('Displacement y-direction (px)')
    plt.title('Displacement y-direction Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    
    
    frames_pos_list_y = []
    for list_i in groups_avr_y :
        frames_pos_list_y.append (next((index for index, value in enumerate(list_i) 
                                           if value > threshold_disp_curves ), None))
        
    # Plotting the distances
    plt.figure(figsize=(20, 15))
    
    for group in range(len(groups_avr_y)):
        
        plt.plot(groups_avr_y[group], label = f"group {group+1}")
        
        if  isinstance(frames_pos_list_y[group], int) :
            plt.axvline(x=frames_pos_list_y[group], color='r', linestyle='--', linewidth=1)  # Draw vertical line
            plt.text(frames_pos_list_y[group], text_pos +group , f"{frames_pos_list_y[group]}", verticalalignment='bottom', color='r')  # Add text annotation
    
    
    plt.axhline(y = threshold_disp_curves, color='r', linestyle='--', linewidth=1)  # Draw vertical line
    
    plt.xlabel('Frame No.')
    plt.ylabel('Displacement since First Frame')
    plt.title('Displacement since First Frame Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    ##############################################################################
    ################################ x displacements 
    
    
    plt.figure(figsize=(25, 15))
    # calculate the absolute distances
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
            distances = [calculate_distancex(pre_row, pre_col, new_row, new_col) for new_row, new_col in zip(data['new_row'], data['new_col'])]
            plt.plot(distances)   #  label=str(sheet_name)

    
    plt.axvline(x=compression_start, color='brown', linestyle='--', label= f'Compression start {compression_start}', linewidth = 4)
    plt.axvline(x=compression_end, color='black', linestyle='--', label= f'Compression end {compression_end}', linewidth = 4)
    # plt.axvline(x=compression_end, color='brown', linestyle='--', label= f'Compression end {compression_end}', linewidth = 4)

    plt.xlabel('Frame number')
    plt.ylabel('Displacement in pixels')
    plt.title('Horizontal displacement of nodes over the frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    distancesx_list = []
    # Plotting the x distances
    plt.figure(figsize=(20, 15))
    
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
            distancesx = [calculate_distancex(pre_row, pre_col, new_row, new_col) for new_row, new_col in zip(data['new_row'], data['new_col'])]
            
            # distancesx = distancesx[:compression_end]
    
            distancesx_list.append(distancesx)
            
            color_number = int(sheet_name) % 4   
            plt.plot(distancesx, alpha=alpha_transparency, color = f'C{color_number}')   #  label=str(sheet_name)
    
    
    groups_avr_x = node_groups_average(distancesx_list)

    
    for group in range(len(groups_avr_x)):
        plt.plot(groups_avr_x[group], label = f"group {group+1}", linewidth = 3)
    
    # plot vertical lines indicating the position of peaks on the first axis
    for ii in range(1, peak_df.shape[0]-1):
        
        color_i = 3+ii
        # variable to lable the peaks with letters A, B, C and D
        letter = chr(ord('A')+ii-1)
        # Draw the vertical line at the mapped value

        peak_i_pos_Frame = peak_df['Frame_n'][ii]
        
        plt.axvline(x=peak_i_pos_Frame, color= f'C{color_i}', linestyle='--', label=f'Peak {letter} at {peak_i_pos_Frame}', linewidth = 3)
        
    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x= compression_start , color='black', linestyle='--', label= f'Compression start {compression_start}', linewidth = 3)

    # # Draw a vertical line indicating the end of compression.
    plt.axvline(x=compression_end, color='black', linestyle='--', label= f'Compression end {compression_end}', linewidth = 3)
    
    
    # Adding more ticks on the displacement axis in mm
    # x_ticks = np.arange(0, 215, 10) 
    # plt.xticks(x_ticks)
    
    
    plt.xlabel('Frame No.')
    plt.ylabel('Displacement x-direction (px)')
    plt.title('Displacement x-direction Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    frames_pos_list_x = []
    for list_i in groups_avr_x :
        frames_pos_list_x.append (next((index for index, value in enumerate(list_i) 
                                           if value > threshold_disp_curves or value < -threshold_disp_curves ), None))
    
    
    # Plotting the distances
    plt.figure(figsize=(20, 15))
    
    
    for group in range(len(groups_avr_x)):
        
        plt.plot(groups_avr_x[group], label = f"group {group+1}")
        plt.axvline(x=frames_pos_list_x[group], color='r', linestyle='--', linewidth=1)  # Draw vertical line
        plt.text(frames_pos_list_x[group], text_pos+group , f"{frames_pos_list_x[group]}", verticalalignment='bottom', color='r')  # Add text annotation
    
    plt.axhline(y = threshold_disp_curves, color='r', linestyle='--', linewidth=1)  # Draw vertical line
    plt.axhline(y = -threshold_disp_curves, color='r', linestyle='--', linewidth=1)  # Draw vertical line
    
    
    plt.xlabel('Frame No.')
    plt.ylabel('Displacement since First Frame')
    plt.title('Displacement since First Frame Over the Frames')
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()
    
    
    
    output_data = []

    for sheet_name, data in excel_data.items():
        if not data.empty:
            pre_row, pre_col = get_reference_pre_coords(data, Ref_frame)
            if pre_row is None:
                # skip this node if it has no reference frame
                continue

            # we still keep the length checks; this is positional,
            # but now at least Ref_frame is safe
            distance_clustering_Frame  = (
                calculate_distance(pre_row, pre_col,
                    data.iloc[clustering_Frame_1]['new_row'], data.iloc[clustering_Frame_1]['new_col'])
                if len(data) > clustering_Frame_1 else None
            )
            distance_clustering_Framex = (
                calculate_distancex(pre_row, pre_col,
                    data.iloc[clustering_Frame_1]['new_row'], data.iloc[clustering_Frame_1]['new_col'])
                if len(data) > clustering_Frame_1 else None
            )
            distance_clustering_Framey = (
                calculate_distancey(pre_row, pre_col,
                    data.iloc[clustering_Frame_1]['new_row'], data.iloc[clustering_Frame_1]['new_col'])
                if len(data) > clustering_Frame_1 else None
            )

            distance_Frame_2  = (
                calculate_distance(pre_row, pre_col,
                    data.iloc[Frame_2]['new_row'], data.iloc[Frame_2]['new_col'])
                if len(data) > Frame_2 else None
            )
            distance_Frame_2x = (
                calculate_distancex(pre_row, pre_col,
                    data.iloc[Frame_2]['new_row'], data.iloc[Frame_2]['new_col'])
                if len(data) > Frame_2 else None
            )
            distance_Frame_2y = (
                calculate_distancey(pre_row, pre_col,
                    data.iloc[Frame_2]['new_row'], data.iloc[Frame_2]['new_col'])
                if len(data) > Frame_2 else None
            )

            output_data.append((
                sheet_name,
                distance_clustering_Frame,  distance_clustering_Framex, distance_clustering_Framey,
                distance_Frame_2,           distance_Frame_2x,          distance_Frame_2y
            ))
            
    # Create a DataFrame from the collected data
    output_df = pd.DataFrame(output_data, columns=['Node', f'Displacement at {clustering_Frame_1}',  
                                                   f'Displacement at {clustering_Frame_1} - y',  
                                                   f'Displacement at {clustering_Frame_1} - x', 
                                                   f'Displacement at {Frame_2}',  
                                                   f'Displacement at {Frame_2} - y',  
                                                   f'Displacement at {Frame_2} - x'])
    
    
    # Save the DataFrame to a new Excel file
    output_file_path = f'{test_folder}/nodes_in_two_frames_{test_info}.xlsx'
    output_df.to_excel(output_file_path, index=False)
    
    # ==========================================================
    # Export per-node displacements (dx, dy) vs frame to CSV
    # ==========================================================

    disp_rows = []

    for sheet_name, data in excel_data.items():
        if data.empty:
            continue

        pre_row, pre_col = get_reference_pre_coords(data, Ref_frame)
        if pre_row is None:
            continue

        try:
            node_id = int(sheet_name)
        except ValueError:
            # skip non-numeric sheet names if any
            continue

        pre_row0 = data.iloc[0]['pre_row']
        pre_col0 = data.iloc[0]['pre_col']
        
        for frame, new_r, new_c in zip(data['frame_number'], data['new_row'], data['new_col']):
            dx = calculate_distancex(pre_row0, pre_col0, new_r, new_c)
            dy = calculate_distancey(pre_row0, pre_col0, new_r, new_c)

            disp_rows.append({
                "node":         node_id,
                "frame_number": int(frame),
                "dx_pixels":    float(dx),
                "dy_pixels":    float(dy),
            })

    if disp_rows:
        disp_df = pd.DataFrame(disp_rows)
        disp_csv_path = os.path.join(
            test_folder,
            f"node_displacements_vs_frame_{test_info}.csv",
        )
        disp_df.to_csv(disp_csv_path, index=False)
        print(f"Saved node displacements to {disp_csv_path}")
    else:
        print("No displacement data to export (disp_rows is empty).")
    
    
    
    
    
    
    
    #####################################################################################
    # dendrogram clustering based on the absolute displacements
    
    # Initialize a list to collect the distances
    distances_clustering_Frame = []
    
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
    
            # Collect distance at position clustering_Frame_1 if it exists
            if len(data) > clustering_Frame_1:
                distance_clustering_Frame = calculate_distance(pre_row, pre_col, data.iloc[clustering_Frame_1]['new_row'], data.iloc[clustering_Frame_1]['new_col'])
                distances_clustering_Frame.append((sheet_name, distance_clustering_Frame))
    
    
    # Create a DataFrame from the collected distances
    distances_df = pd.DataFrame(distances_clustering_Frame, columns=['Node', 'distance_clustering_Frame']).dropna()
    
    # Extract the distance values for clustering
    distance_values = distances_df['distance_clustering_Frame'].values.reshape(-1, 1)
    
    # Perform hierarchical clustering
    linked = linkage(distance_values, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(15, 10))
    
    min_distance_for_3_clusters = min(linked[-3:, 2])  # Extracting the maximum distance among the last (n-1) merges
    min_distance_for_7_clusters = min(linked[-7:, 2])  # Extracting the maximum distance among the last (n-1) merges
    
    # Define the distance threshold
    min_distance_lim = min_distance_for_3_clusters-1  # Adjust this to the desired distance for annotating clusters
    
    dendrogram_result = dendrogram(linked,
                                   orientation='top',
                                   labels=distances_df['Node'].values,
                                   show_leaf_counts=True,
                                   color_threshold=min_distance_lim)
    
    # dendrogram(Z,
    #             orientation='top',
    #             labels=distances_df['Node'].values,
    #             distance_sort='descending',
    #             show_leaf_counts=True)
    
    
    # Get the cluster labels for each data point
    cluster_labels = fcluster(linked, min_distance_lim, criterion='distance')
    
    # Initialize a dictionary to hold lists of components for each cluster
    clusters_abs = {i: [] for i in range(1, max(cluster_labels) + 1)}
    
    # Group data points by their cluster labels
    for idx, cluster in enumerate(cluster_labels):
        clusters_abs[cluster].append(idx)
    
    # Display the cluster components
    for cluster_id, components in clusters_abs.items():
        print(f"Subcluster {cluster_id}: {components}")
        
        
    
    clusters_counter = 1
    
    # Annotate clusters at the specified distance
    for i, d, c in zip(dendrogram_result['icoord'], dendrogram_result['dcoord'], dendrogram_result['ivl']):
        if max(d) >= min_distance_for_3_clusters:  # Only annotate clusters that merge above the distance threshold
            x = 0.5 * sum(i[1:3])  # Midpoint of the cluster on the x-axis
            y = max(d)  # Distance where the clusters are merged
            plt.text(x, y + 0.1, f'C{clusters_counter}', fontsize=20, color='red', ha='center')
            clusters_counter += 1
    
    clusters_counter = 1
    # Annotate clusters at the specified distance
    for i, d, c in zip(dendrogram_result['icoord'], dendrogram_result['dcoord'], dendrogram_result['ivl']):
        if max(d) < min_distance_for_3_clusters and max(d) >= min_distance_for_7_clusters:  # Only annotate clusters that merge above the distance threshold
            x = 0.5 * sum(i[1:3])  # Midpoint of the cluster on the x-axis
            y = max(d)  # Distance where the clusters are merged
            plt.text(x, y + 0.1, f'SC{clusters_counter}', fontsize=20, color='red', ha='center')
            clusters_counter += 1
    
            
    plt.axhline(y=min_distance_lim, color='gray', linestyle='--', label=f'Distance Threshold = {min_distance_lim}')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.xlabel('Nodes')
    plt.ylabel('Distance')
    plt.ylim([0,40])
    plt.title(f'Dendrogram of Nodes based on the absolute value of Displacements at Position {clustering_Frame_1}')
    plt.show()
    
    
    # Elbow method for finding the optimal number of clusters
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(distance_values)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    plt.show()
    plt.close()
    
    
    
    clustering_image_1 = cv2.imread(f"{frames_folder}/frame_{0:04d}.jpg", cv2.IMREAD_GRAYSCALE) 
    clustering_graph = Loaded_Graphs_List[0]
    
    # update nodes and edges after relabelling and filtering
    nodes = clustering_graph.nodes()    
    
    # update the previous array fo the next frame
    # initiate an array of zeros with number of rows equal to the number of nodes
    # and with 3 columns for [node_name , node_row_position , nodes_column_position]
    nodes_coord_array = np.zeros( (clustering_graph.number_of_nodes() ,3) ).astype(int)
    # initiate a counter of the node number to 0
    counter = 0 
    
    # for loop to read and store the node information in the array as follows:
    # every row containes: [node_name , node_row_position , nodes_column_position]
    for node in nodes:
        node_info_list = [node, nodes[node]['o'][0] , nodes[node]['o'][1]]
        nodes_coord_array[counter,:] = np.array(node_info_list)
        # increment the counter
        counter +=1

    # Convert ndarray to DataFrame
    df_node_pos = pd.DataFrame(nodes_coord_array)
    # Set the first column as the index
    df_node_pos.set_index(0, inplace=True)
    node_pos_dict = {index: (row[2], row[1]) for index, row in df_node_pos.iterrows()}
    
    
    fig_s = 22
    node_s = 300
    font_s = 13

    # # Create a color map for the nodes
    # node_colors = []

    # # Assign colors to each node based on its subgroup
    # for node in clustering_graph.nodes():
    #     if node in clusters_abs[1] :
    #         node_colors.append('orange')  # Color for subgroup 1
    #     elif node in clusters_abs[2]:
    #         node_colors.append('green')  # Color for subgroup 2
    #     elif node in clusters_abs[3]:
    #         node_colors.append('red')  # Color for subgroup 2
    #     elif node in clusters_abs[4]:
    #         node_colors.append('purple')  # Color for subgroup 2
            
            
    
    # # Print the binary image
    # plt.figure(figsize= (fig_s, fig_s))
    # plt.imshow(clustering_image_1, "gray", vmin=0, vmax=255)

    # # draw the graph
    # nx.draw(clustering_graph, node_pos_dict , with_labels=True,
    #         node_color= node_colors, node_size=node_s, edge_color="yellow",
    #         font_color="white",font_size=font_s, 
    #         font_weight="bold", width=2)

    # # title and show
    # plt.title(f'Node groups based on Dendrogram of absolute displacement at Frame {clustering_Frame_1}')
    # plt.show()
    
    
    
    # Create a color map for the nodes
    node_colors = []

    # Assign colors to each node based on its subgroup
    for node in clustering_graph.nodes():
        if node in clusters_abs[1] or node in clusters_abs[2] :
            node_colors.append('blue')  # Color for subgroup 1
        else:
            node_colors.append('red')  # Color for subgroup 2
    
    # Print the binary image
    plt.figure(figsize= (fig_s, fig_s))
    plt.imshow(clustering_image_1, "gray", vmin=0, vmax=255)

    # draw the graph
    nx.draw(clustering_graph, node_pos_dict , with_labels=True,
            node_color= node_colors, node_size=node_s, edge_color="yellow",
            font_color="white",font_size=font_s, 
            font_weight="bold", width=2)

    # title and show
    plt.title(f'Node groups based on Dendrogram of absolute displacement at Frame {clustering_Frame_1}')
    plt.show()


    
    
    ###############################################################################
    # dendrogram clustering based on the y displacements
    
    # Initialize a list to collect the distances
    distances_clustering_Frame = []
    
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
    
            # Collect distance at position clustering_Frame_1 if it exists
            if len(data) > clustering_Frame_2:
                distance_clustering_Frame = calculate_distancey(pre_row, pre_col, data.iloc[clustering_Frame_2]['new_row'], data.iloc[clustering_Frame_2]['new_col'])
                distances_clustering_Frame.append((sheet_name, distance_clustering_Frame))
    
    
    # Create a DataFrame from the collected distances
    distances_df = pd.DataFrame(distances_clustering_Frame, columns=['Node', 'distance_clustering_Frame']).dropna()
    
    # Extract the distance values for clustering
    distance_values = distances_df['distance_clustering_Frame'].values.reshape(-1, 1)
    
    # Perform hierarchical clustering
    linked = linkage(distance_values, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(15, 10))
    
    min_distance_for_3_clusters = min(linked[-3:, 2])  # Extracting the maximum distance among the last (n-1) merges
    min_distance_for_2_clusters = min(linked[-2:, 2])  # Extracting the maximum distance among the last (n-1) merges
    
    # Define the distance threshold
    min_distance_lim = min_distance_for_2_clusters+4  # Adjust this to the desired distance for annotating clusters
    
    dendrogram_result = dendrogram(linked,
                                   orientation='top',
                                   labels=distances_df['Node'].values,
                                   show_leaf_counts=True,
                                   color_threshold=min_distance_lim)
    
    
    # Get the cluster labels for each data point
    cluster_labels = fcluster(linked, min_distance_lim, criterion='distance')
    
    # Initialize a dictionary to hold lists of components for each cluster
    clusters_y = {i: [] for i in range(1, max(cluster_labels) + 1)}
    
    # Group data points by their cluster labels
    for idx, cluster in enumerate(cluster_labels):
        clusters_y[cluster].append(idx)
    
    # Display the cluster components
    for cluster_id, components in clusters_y.items():
        print(f"Subcluster {cluster_id}: {components}")
        
    clusters_counter = 1
    
    # Annotate clusters at the specified distance
    for i, d, c in zip(dendrogram_result['icoord'], dendrogram_result['dcoord'], dendrogram_result['ivl']):
        if max(d) >= min_distance_for_3_clusters:  # Only annotate clusters that merge above the distance threshold
            x = 0.5 * sum(i[1:3])  # Midpoint of the cluster on the x-axis
            y = max(d)  # Distance where the clusters are merged
            plt.text(x, y + 0.1, f'C{clusters_counter}', fontsize=20, color='red', ha='center')
            clusters_counter += 1
    
    
    plt.axhline(y=min_distance_lim, color='gray', linestyle='--', label=f'Distance Threshold = {min_distance_lim}')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0,40])
    plt.xlabel('Nodes')
    plt.ylabel('Distance')
    plt.title(f'Dendrogram of Nodes based on y-Displacement at Position {clustering_Frame_2}')
    plt.show()
    
    
    
    
    # Elbow method for finding the optimal number of clusters
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(distance_values)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    # plt.show()
    plt.close()
    
    
    
    # Create a color map for the nodes
    node_colors = []

    # Assign colors to each node based on its subgroup
    for node in clustering_graph.nodes():
        if node in clusters_y[1] :
            node_colors.append('orange')  # Color for subgroup 1
        else:
            node_colors.append('green')  # Color for subgroup 2
    
    # Print the binary image
    plt.figure(figsize= (fig_s, fig_s))
    plt.imshow(clustering_image_1, "gray", vmin=0, vmax=255)

    # draw the graph
    nx.draw(clustering_graph, node_pos_dict , with_labels=True,
            node_color= node_colors, node_size=node_s, edge_color="yellow",
            font_color="white",font_size=font_s, 
            font_weight="bold", width=2)

    # title and show
    plt.title(f'Node groups based on Dendrogram of y-displacement at Frame {clustering_Frame_2}')
    plt.show()
    
    
    
    
    ###############################################################################
    # dendrogram clustering based on the x displacements
    
    # Initialize a list to collect the distances
    distances_clustering_Frame = []
    
    for sheet_name, data in excel_data.items():
        if not data.empty:
            # Ensure pre_row and pre_col are from the first row of the current sheet
            pre_row, pre_col = data.iloc[0]['pre_row'], data.iloc[0]['pre_col']
    
            # Collect distance at position clustering_Frame_1 if it exists
            if len(data) > clustering_Frame_2:
                distance_clustering_Frame = calculate_distancex(pre_row, pre_col, data.iloc[clustering_Frame_2]['new_row'], data.iloc[clustering_Frame_2]['new_col'])
                distances_clustering_Frame.append((sheet_name, distance_clustering_Frame))
    
    
    # Create a DataFrame from the collected distances
    distances_df = pd.DataFrame(distances_clustering_Frame, columns=['Node', 'distance_clustering_Frame']).dropna()
    
    # Extract the distance values for clustering
    distance_values = distances_df['distance_clustering_Frame'].values.reshape(-1, 1)
    
    # Perform hierarchical clustering
    linked = linkage(distance_values, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(15, 10))
    
    min_distance_for_3_clusters = min(linked[-3:, 2])  # Extracting the maximum distance among the last (n-1) merges
    min_distance_for_2_clusters = min(linked[-2:, 2])  # Extracting the maximum distance among the last (n-1) merges
    
    # Define the distance threshold
    min_distance_lim = min_distance_for_2_clusters+4  # Adjust this to the desired distance for annotating clusters
    
    # Get the cluster labels for each data point
    cluster_labels = fcluster(linked, min_distance_lim, criterion='distance')
    exch = False;
    if cluster_labels[0] == 1:
        exch = True;
        for ind, ii in enumerate(cluster_labels):
            if ii == 1:
                cluster_labels[ind] = 2;
            else:
                cluster_labels[ind] = 1;
                
        # Create a custom color palette for reversing colors when exch = 1
    

    dendrogram_result = dendrogram(linked,
                               orientation='top',
                               labels=distances_df['Node'].values,
                               show_leaf_counts=True,
                               color_threshold=min_distance_lim)
        
        
    
    # Initialize a dictionary to hold lists of components for each cluster
    clusters_x = {i: [] for i in range(1, max(cluster_labels) + 1)}
    
    # Group data points by their cluster labels
    for idx, cluster in enumerate(cluster_labels):
        clusters_x[cluster].append(idx)
    
    # Display the cluster components
    for cluster_id, components in clusters_x.items():
        print(f"Subcluster {cluster_id}: {components}")
        
    clusters_counter = 1
    
    # Annotate clusters at the specified distance
    for i, d, c in zip(dendrogram_result['icoord'], dendrogram_result['dcoord'], dendrogram_result['ivl']):
        if max(d) >= min_distance_for_3_clusters:  # Only annotate clusters that merge above the distance threshold
            x = 0.5 * sum(i[1:3])  # Midpoint of the cluster on the x-axis
            y = max(d)  # Distance where the clusters are merged
            if exch:
                plt.text(x, y + 0.1, f'C{-clusters_counter+3}', fontsize=20, color='red', ha='center')
            else:
                plt.text(x, y + 0.1, f'C{clusters_counter}', fontsize=20, color='red', ha='center')
            clusters_counter += 1
    
            
    plt.axhline(y=min_distance_lim, color='gray', linestyle='--', label=f'Distance Threshold = {min_distance_lim}')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([0,40])
    plt.xlabel('Nodes')
    plt.ylabel('Distance')
    plt.title(f'Dendrogram of Nodes based on x-Displacements at Position {clustering_Frame_2}')
    plt.show()
    
    
    
    # Elbow method for finding the optimal number of clusters
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(distance_values)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)
    # plt.show()
    plt.close()
    
    
    
    
    # -------------------------------------------------------------
    # Color map for nodes (x-displacement clustering, Frame_2)
    # -------------------------------------------------------------
    # Use an explicit ordered list of nodes, so colors and positions
    # are guaranteed to have the same length and order.
    nodes_list = list(clustering_graph.nodes())
    
    node_colors = []
    if exch:
        # if exch is True, invert the coloring logic
        for node in nodes_list:
            if node not in clusters_x[1]:
                node_colors.append("orange")
            else:
                node_colors.append("green")
    else:
        for node in nodes_list:
            if node not in clusters_x[1]:
                node_colors.append("green")
            else:
                node_colors.append("orange")
    
    # safety check (optional, but useful for debugging)
    print("[DEBUG] x-clustering: #nodes =", len(nodes_list),
          ", #colors =", len(node_colors))
    
    # Print the binary image
    plt.figure(figsize=(fig_s, fig_s))
    plt.imshow(clustering_image_1, "gray", vmin=0, vmax=255)
    
    # draw the graph, using the explicit nodelist
    nx.draw(
        clustering_graph,
        node_pos_dict,
        nodelist=nodes_list,
        with_labels=True,
        node_color=node_colors,
        node_size=node_s,
        edge_color="yellow",
        font_color="white",
        font_size=font_s,
        font_weight="bold",
        width=2,
    )
    
    # title and show
    plt.title(
        f"Node groups based on Dendrogram of x-displacement at Frame {clustering_Frame_2}"
    )
    plt.show()

    
    # # Plot the nodes corresponding to the pink curves, so 6, 16, 26, 36 
    # to check the displacement of node 6 that was wrongly clustered
    # plt.figure(figsize=(15, 10))
    # plt.plot(distancesx_list[6])
    # plt.plot(distancesx_list[16])
    # plt.plot(distancesx_list[26])
    # plt.plot(distancesx_list[36])
    # plt.xlabel('Frames')
    # plt.ylabel('disp')
    # plt.title('Nodes 6, 16, 26, 36')
    # plt.grid(True)
    # plt.show()
    
    
    subclusters = { 1 : list(set(clusters_x[2]) & set(clusters_y[2])),
                    2 : list(set(clusters_x[1]) & set(clusters_y[1])),
                    3 : list(set(clusters_x[2]) & set(clusters_y[1])),
                    4 : list(set(clusters_x[1]) & set(clusters_y[2]))
                    }
    
    
    # set_subcluster_1 = list(set(clusters_x[2]) & set(clusters_y[2]))
    # set_subcluster_2 = set(clusters_x[1]) & set(clusters_y[2])
    # set_subcluster_3 = set(clusters_x[2]) & set(clusters_y[1])
    # set_subcluster_4 = set(clusters_x[2]) & set(clusters_y[2])

    
    # Create a color map for the nodes
    node_colors = []
    
    # Assign colors to each node based on its subgroup
    for node in clustering_graph.nodes():
        if node in subclusters[1] :
            node_colors.append('blue')  # Color for subgroup 1
        elif node in subclusters[2]:
            node_colors.append('orange')  # Color for subgroup 2
        elif node in subclusters[3]:
            node_colors.append('green')  # Color for subgroup 2
        elif node in subclusters[4]:
            node_colors.append('red')  # Color for subgroup 2
    
    # Print the binary image
    plt.figure(figsize= (fig_s, fig_s))
    plt.imshow(clustering_image_1, "gray", vmin=0, vmax=255)
    
    # draw the graph
    nx.draw(clustering_graph, node_pos_dict , with_labels=True,
            node_color= node_colors, node_size=node_s, edge_color="yellow",
            font_color="white",font_size=font_s, 
            font_weight="bold", width=2)
    
    # title and show
    plt.title(f'Node groups based on Dendrogram of x- and y- displacements at Frame {clustering_Frame_2}')
    plt.show()
        





    # plotting with 2 x axis: deformation in mm and the corresponding Frames number 
    fig, ax1 = plt.subplots(figsize=(24, 12))

    # Plot with the first x-axis (crush in mm)
    ax1.plot(df_unique['Crush (mm)'], df_unique['Force (N)'], color='b')
    ax1.set_xlabel('Deformation (mm)')
    ax1.set_ylabel('Load (N)')
    ax1.set_xlim(0,3)
    # ax1.set_xlim(-0.2 ,2.2)

    ax1.set_ylim(0,800)
    ax1.set_title('Load-Deformation curve')

    # Create the frames axis
    ax2 = ax1.twiny()


    ax2.set_xlabel('Frame number')
    # Synchronize the 2 axis
    ax2.set_xlim(ax1.get_xlim())

    # Set  ticks and labels for the second x-axis
    ax2.set_xticks(df_unique['Crush (mm)'][:: math.floor(50/fps_div)])
    ax2.set_xticklabels(df_unique['Frame_n'][:: math.floor(50/fps_div)])



    # define a list of colors for the plot
    color_list = ['b', 'g', 'r', 'y', 'm','pink']

    # plot vertical lines indicating the position of peaks on the first axis
    for ii in range(1, peak_df.shape[0]-1):
        
        # variable to lable the peaks with letters A, B, C and D
        letter = chr(ord('A')+ii-1)
        # Draw the vertical line at the mapped value
        
        peak_i_pos_crush = peak_df['Crush (mm)'][ii]

        peak_i_pos_Frame = peak_df['Frame_n'][ii]
        
        ax1.axvline(x= peak_i_pos_crush, color=color_list[ii], linestyle='--', label=f'Peak {letter} at {peak_i_pos_Frame}')
        
        
    # # Draw a vertical line indicating the end of compression.
    ax1.axvline(x=df_unique['Crush (mm)'].iloc[-1], color='brown', linestyle='--', label='End')
    
    
    prediction_all_peak_pos = frames_pos_list_x
    
    prediction_all_peak_pos.sort()
    
    for position in prediction_all_peak_pos :
        
        # avoid errors coming from nan values in listes in case peaks were not detected
        if isinstance(position, int) :
            
            # pos_corresponding_crush = (position - compression_start) / coeff
            
            df_peak_i = df_unique[df_unique['Frame_n'] == position].reset_index(drop=True)
            pos_corresponding_frame = df_peak_i.loc[0,'Frame_n']
            pos_corresponding_crush = df_peak_i.loc[0,'Crush (mm)']
            ax1.axvline(x=pos_corresponding_crush, color='C9', linestyle='--', linewidth=2, label=f'Prediction at {pos_corresponding_frame}')


    df_peak_i = df_unique[df_unique['Frame_n'] == frame_pos_peak_D].reset_index(drop=True)
    pos_corresponding_frame = df_peak_i.loc[0,'Frame_n']
    pos_corresponding_crush = df_peak_i.loc[0,'Crush (mm)']
    ax1.axvline(x=pos_corresponding_crush, color='C9', linestyle='--', linewidth=2, label=f'Buckling start at {pos_corresponding_frame}')

    df_peak_i = df_unique[df_unique['Frame_n'] == frame_buckling_end].reset_index(drop=True)
    pos_corresponding_frame = df_peak_i.loc[0,'Frame_n']
    pos_corresponding_crush = df_peak_i.loc[0,'Crush (mm)']
    ax1.axvline(x=pos_corresponding_crush, color='C9', linestyle='--', linewidth=2, label=f'Buckling end at {pos_corresponding_frame}')

    
    
    # Combine legends from both axes
    ax1.legend(fontsize=20)
    # Adding more ticks on the displacement axis in mm
    x_ticks = np.arange(0, 3, 0.1) 
    ax1.set_xticks(x_ticks)
    ax1.grid("on")
    plt.show()
    plt.close()


#%% Board modeling in a selected frame under the given load conditions there 
    # Stress analysis under given load and boundary conditions


def board_mod(board_modeling):
    
    def get_effective_modulus(E_MD, E_ZD, x_start, y_start, x_end, y_end):
        dx = x_end - x_start
        dy = y_end - y_start
        theta = np.arctan2(dy, dx)  # angle with respect to horizontal
        E_theta = E_MD * (np.cos(theta)**2) + E_ZD * (np.sin(theta)**2)
        return E_theta

    if board_modeling :
        
        # geometry and material constants
        
        # geometry constants of the fluting medium
        # hight
        h_flute = 0.2                 # in mm
        # cross-section area of the paper sheet
        A_flute = h_flute*b           # in mm^2
        # moment of inertia of the paper sheet
        I_flute = b*(h_flute**3) /12  # in mm^4
        
        # geometry constants of the liners
        h_liner = 0.2                 # in mm
        A_liner = h_liner*b           # in mm^2
        I_liner = b*(h_liner**3) /12  # in mm^4
        
        # geometry constants of the contact region between Liners and fluting medium.
        h_both = h_flute + h_liner
        h_both = round(h_both,4)
        A_both = h_both *b
        I_both = b*(h_both**3) /12
        
        # E-modulus in MD of the paper sheets
        E_ax = 2500         # in MPa = N/mm^2
        # E-Modul in ZD of the paper sheets
        E_b  = E_ax/200     # in MPa = N/mm^2
        
        ## E_b  = 200      # in MPa = N/mm^2  ###### value took from an artical (not sure for board or paper)
        # comparison to a thesis-paper where I found that E/b = 8 to 11
        c = E_b/b
        
        # Axial stiffness
        EA_flute = E_ax* A_flute
        EA_liner = E_ax* A_liner
        EA_both  = E_ax* A_both
        
        # Bending stiffness
        EI_flute = E_b*I_flute
        EI_liner = E_b*I_liner
        EI_both  = E_b*I_both
        
        
            
        # sorted lists of upper and lower liner nodes
        upper_liner_nodes_list = sorted(list(node_pos_dict_upper_liner_0.keys()))
        lower_liner_nodes_list = sorted(list(node_pos_dict_lower_liner_0.keys()))
    
        
        
        if model_elasticity :
            
            # elasticity for fps_div = 10  is between frames 39 and 43
            model_ref_frame = 113
            model_end_frame = 118
            
    
        elif model_all:
            # elasticity for fps_div = 10  is between frames 39 and 43
            model_ref_frame = compression_start
            model_end_frame = compression_end-1
            
    
        
    
        # read the index corresponding to this frame 
        ref_frame_df_index = df_unique[df_unique['Frame_n'] == model_ref_frame].index[0]
    
    
        # save the graph data corresponding to the chosen frame to a variable
        ref_graph = Loaded_Graphs_List[model_ref_frame]
        
        
        # read the graph data in millimeters with the function graph_data_mm
        (node_pos_dict_i, 
        node_pos_dict_upper_liner_i, 
        node_pos_dict_lower_liner_i,
        nodes_list_i,
        thickness_estimation_i          ) = graph_data_mm(ref_graph, pixel_mm)
        
        
        # attempt to verify if the current graph is equivalent to the initial graph
        # if  set (nodes_list_0) == set (nodes_list_i) and set(edges_list_i) == set (edges_list_0):
        #     print('Nodes and Edges are conserved in this frame')
        # else:
        #     print('Nodes or Edges are not conserved in this frame')
        
        
        ######################################################################
        ######### old method to model the board in peak positions or any specific frame
        
        # # select a peak or frame number. Peak numbers have a priority it both values are given
        # Frame_number = None
        # peak_number = None
        
        # if peak_number is not None:
        #     # read the frame number corrsponding to this peak from the peaks dataframe
        #     Frame_number = peak_df.iloc[peak_number]['Frame_n'].astype(int)
        #     # read the load corrsponding to this peak
        #     load_N = peak_df.iloc[peak_number]['Force (N)']
        #     # print(f'Frame number: {Frame_number}')
        # elif Frame_number is not None:
        #     # read the index corresponding to this frame 
        #     frame_df_index = df_unique[df_unique['Frame_n'] == Frame_number].index[0]
        #     # extract the data corresponding to this frame
        #     df_i = df_unique.iloc[[frame_df_index]]
        #     # read the load corresponding to the next frame
        #     load_N = df_unique.loc[frame_df_index]['Force (N)']
        #     # print the frame number and the corresponding data
        #     # print(Frame_number)
        #     print(df_i)
        #     print(load_N)
        # else:
        #     print('specify a frame or a peak number where to model the corrugated board')
        
        ######################################################################
    
    
        
        
    
        Frame_number = model_ref_frame
        
        # Create the structure system
        ss = SystemElements()
        
        # initiate a counter for all the beam elements to add to the struchture
        element_count = 0
        # initiate a list for the elements constituting the flute
        flute_elements_list = []
        contact_elements_list = []
        
        # Add flute elements and nodes to the structure
        for index, node in enumerate(nodes_list_i[:-1]) :
            
            # take the node labels limiting the element
            node_start = node
            node_end = node+1
            
            # read the nodes' positions from the dictionaries
            x_start, y_start = node_pos_dict_i[node_start]
            x_end, y_end = node_pos_dict_i[node_end]
            
            # if the element to be added is a flute column then add a flute element
            if  ((node_start in upper_liner_nodes_list and node_end in lower_liner_nodes_list )) or ((node_start in lower_liner_nodes_list and node_end in upper_liner_nodes_list ))  :
                E_theta = get_effective_modulus(E_ax, E_b, x_start, y_start, x_end, y_end)
                EA = E_theta * A_flute
                EI = E_theta * I_flute
                ss.add_element(location=[[x_start, y_start], [x_end, y_end]], EA=EA, EI=EI)
                            
                                # this is how to add plastic hinges in the node positions
                                # mp={1 : 0.05, 2 : 0.05}
                                
                # increment the element counter
                element_count +=1
                # add the element number to a list
                flute_elements_list.append(element_count)
                            
                            
            # else if the element to be added is a contact region beween the fluting and
            # the liners then add a combination of both fluting and liner. 
            else:
                
                E_theta = get_effective_modulus(E_ax, E_b, x_start, y_start, x_end, y_end)
                EA = E_theta * A_both
                EI = E_theta * I_both
                ss.add_element(location=[[x_start, y_start], [x_end, y_end]], EA=EA, EI=EI)
                
                # increment the element counter
                element_count +=1
                # add the element number to a list
                contact_elements_list.append(element_count)
        
        
        # fig = ss.show_structure(scale=0.55,show=False)
        # plt.title(f'Flute structure of the currugated board in frame number {Frame_number}')
        # plt.xlabel('Distance in mm')
        # plt.ylabel('Distance in mm')
        # plt.show()
        
        
        ######## Do nearly the same to add the missing upper liner elements
        
        
        upper_liner_elements_list = []
        # Add elements to the structure
        for index, node in enumerate(upper_liner_nodes_list[:-1]) :
                
            node_start = node
            node_end = upper_liner_nodes_list[index+1]
            x_start, y_start = node_pos_dict_i[node_start]
            x_end, y_end = node_pos_dict_i[node_end]
            
            # elements of the contact regions between fluting and liner  were already added
            # Here add only the missing segments: when the node labels of the element are not
            # directly successive integers.
            if node+1 != upper_liner_nodes_list[index+1]:
                
                E_theta = get_effective_modulus(E_ax, E_b, x_start, y_start, x_end, y_end)
                EA = E_theta * A_liner
                EI = E_theta * I_liner
                ss.add_element(location=[[x_start, y_start], [x_end, y_end]], EA=EA, EI=EI)
            
            element_count +=1      
            upper_liner_elements_list.append(element_count)
            
                
        ######## Do exactly the same to add the missing lower liner elements
        
        lower_liner_elements_list = []
        # Add elements to the structure
        for index, node in enumerate (lower_liner_nodes_list[:-1]) :
        
            node_start = node
            node_end = lower_liner_nodes_list[index+1]
            x_start, y_start = node_pos_dict_i[node_start]
            x_end, y_end = node_pos_dict_i[node_end]
            
            # elements of the contact regions between fluting and liner  were already added
            # Here add only the missing segments: when the node labels of the element are not
            # directly successive integers.
            if node+1!= lower_liner_nodes_list[index+1]:
                
                ss.add_element(location=[[x_start, y_start], [x_end, y_end]],
                                EA = EA_liner , EI = EI_liner)
            
            element_count +=1
            lower_liner_elements_list.append(element_count)
        
        
        # Define the colors for different plot components
        plot_colors = {
            "element_number": "g",
            "node_number": "r" }
        # Use the change_plot_colors method
        ss.change_plot_colors(plot_colors)
        
        
        # plot the structure
        fig = ss.show_structure(scale=0.55,show=False, figsize=(10,6) )
        plt.title(f'Structure of the currugated board in frame number {Frame_number}', fontsize = 20)
        plt.xlabel('Distance in mm', fontsize = 20)
        plt.ylabel('Distance in mm', fontsize = 20)
        # plt.show()
        plt.close()
        
        
        ########################################## Add supports/ boundaries
        # add rotational hinges as boundarie in the lower liner nodes
        ss.add_support_hinged(node_id= lower_liner_nodes_list )
        
        # add rotational roll supports as boundarie in the upper liner nodes
        for ii in upper_liner_nodes_list :
            ss.add_support_roll(node_id = ii, direction=1)
        
        # add internal hinges to all nodes sothat rotation is allowed between the elements
        ss.add_internal_hinge(node_id = nodes_list_i )
        
        
        ######## add roll supports to the lower liner nodes ( in the old model solution)
        # for ii in lower_liner_nodes_list[1:-1] :
        #     ss.add_support_roll(node_id = ii, direction=2)
        
        # plot the structure with boundary conditions
        fig = ss.show_structure(scale=0.55,show=False, verbosity=0, figsize=(10,6) )
        plt.title(f'Structure of the currugated board in frame number {Frame_number}', fontsize = 20)
        plt.xlabel('Distance in mm', fontsize = 20)
        plt.ylabel('Distance in mm', fontsize = 20)
        # plt.show()
        plt.close()
        
        
        
        
        
        
        df_model = df_unique[(df_unique['Frame_n'] >= model_ref_frame) & 
                                  (df_unique['Frame_n'] <= model_end_frame)]
        
        
        
        Crush_ref = df_model.loc[ref_frame_df_index]['Crush (mm)']
        Strain_ref = df_model.loc[ref_frame_df_index]['Strain']
        Load_ref = df_model.loc[ref_frame_df_index]['Force (N)']
        
        Load_list = []
        
        delta_strain_exp_list = []
        strain_model_list = []
        
        delta_y_exp_list = []
        delta_y_model_list = []
        buckling_factor_list = []
        
        
        #########################################################################
        # iteration over the next frames to read load data
        
        for Frame_i in range(model_ref_frame, model_end_frame+1) :
            
            Frame_number = Frame_i
            
            # read the index corresponding to this frame 
            frame_df_index = df_unique[df_unique['Frame_n'] == Frame_number].index[0]
            
            # extract and print the data corresponding to this frame
            # df_i = df_unique.iloc[[frame_df_index]]
            # print(df_i)
            
            # read the load corresponding to the next frame
            load_N = df_unique.loc[frame_df_index+1]['Force (N)'] - Load_ref
            # print the frame number and the corresponding data
            Load_list.append(load_N)
            
            delta_crush = df_unique.loc[frame_df_index+1]['Crush (mm)'] - Crush_ref
            delta_strain = df_unique.loc[frame_df_index+1]['Strain'] - Strain_ref
            
            delta_y_exp_list.append(delta_crush)
            delta_strain_exp_list.append(delta_strain)
            
            
            ########################################## Add point load to the upper liner nodes
            
            # calculate the load per node. Devide the load measured by the total number of upper
            # liner nodes (not the selected nodes for the modeling and plotting)
            Fy_upper_node = round( load_N/ round(graph_0.number_of_nodes()/2-1) ,2)
            
            # define a load manually for the model validation in elasticity.
            # Fy_upper_node = 14.85         # this is the load per node in frame 43 (end of elasticity)
    
            print(f'Frame {Frame_number}  |  Load = {load_N} N  |  load per node = {Fy_upper_node} N')
            
            # Add loads
            ss.point_load(node_id= upper_liner_nodes_list , Fy= -Fy_upper_node)
            
            
            # plot the structure with boundary and load conditions
            fig = ss.show_structure(scale=0.55,show=False, figsize=(10,6))
            plt.title(f'Structure of the currugated board in frame number {Frame_number}',
                      fontsize = 20)
            plt.xlabel('Distance in mm', fontsize = 20)
            plt.ylabel('Distance in mm', fontsize = 20)
            # plt.show()
            plt.close()
            
            struct_fig_size = (10, 8)
            
            ## Solve the system with or without geometrical_non_linear
            ss.solve(geometrical_non_linear=True)
            #geometrical_non_linear=False
            
            
            
            # Convert list of dictionaries to DataFrame
            nodes_disp_df = pd.DataFrame(ss.get_node_displacements(node_id = 0))
            # Filter the DataFrame
            node_upper_liner_disp_df = nodes_disp_df[nodes_disp_df['id'].isin(upper_liner_nodes_list)]
            
            # Set the 'name' column as the index
            node_upper_liner_disp_df.set_index('id', inplace=True)
            # Calculate the column averages
            node_upper_liner_disp_avr = node_upper_liner_disp_df.mean()
            delta_y_model = -node_upper_liner_disp_avr.loc['uy']
            delta_y_model_list.append(delta_y_model)
            
            strain_model = delta_y_model/thickness_estimation_0_mm
            strain_model_list.append(strain_model)
            
    
            
        
            # Convert list of dictionaries to DataFrame
            element_data_df = pd.DataFrame(ss.get_element_results(element_id=0))
            # Filter the DataFrame
            flute_elements_data_df = element_data_df[element_data_df['id'].isin(flute_elements_list)]
            
            flute_elements_data_avr_df = df.mean()
    
            try:
                # Try to add a value to the existing variable
                elements_data_avr_df = pd.concat([elements_data_avr_df,
                                                                    flute_elements_data_avr_df], ignore_index=True)
                
            except NameError:
                # If the variable doesn't exist, create it and assign a value
                
                elements_data_avr_df = flute_elements_data_avr_df
            
            
            buckling_factor_list.append(ss.buckling_factor)
            
            
            
            
            # print('maximum compression force :', round(max(ss.get_element_result_range('axial')),5),  'N  in the element : ', np.argmax(ss.get_element_result_range('axial')) + 1)
            # print('maximum shear force             :', round(max(ss.get_element_result_range('shear')),5),  'N in element      : ', np.argmax(ss.get_element_result_range('shear')) + 1)
            # print('maximum bending moment          :', round(max(ss.get_element_result_range('moment')),5), 'Nm in element     : ', np.argmax(ss.get_element_result_range('moment')) + 1)#
            
            # print('Buckling factor = ',ss.buckling_factor)
            # # print(ss.get_node_results_system(node_id=4)['uy'])
            # disp = ss.get_node_result_range(unit = "uy")
            # print(min(disp))
            # print(sum(disp)/len(upper_liner_nodes_list))
            
            
            
            
            
            
            
            
            # plotting of solve results 
            
            fig_size = 12
            with_fig_size = False
            
            if with_fig_size :
                ## Plot the structure with the loads and boundary conditions
                ss.show_structure(scale=0.55, figsize=(fig_size,fig_size))
                ss.show_reaction_force(scale=0.55, figsize=(fig_size,fig_size))
                ss.show_axial_force(scale=0.55 , factor= 0.1, figsize=(fig_size,fig_size))
                ss.show_bending_moment(scale=0.55, factor= 1, figsize=(fig_size,fig_size))
                ss.show_shear_force(scale=0.55, factor= 1, figsize=(fig_size,fig_size))
                ss.show_displacement(scale=0.55, factor = 1, figsize=(fig_size,fig_size))
            
            else:
                
                # ## Plot the structure with the loads and boundary conditions
                fig = ss.show_structure(scale=0.55,show=False)
                plt.title(f'Load and boundary conditions (frame number {Frame_number})')
                plt.show()
                
                # ## Plot the structure with the loads and boundary conditions
                fig = ss.show_reaction_force(scale=0.55,show=False, figsize= struct_fig_size)
                plt.title(f'Reaction forces (frame number {Frame_number})')
                plt.show()
                
                # ## Plot the structure with the loads and boundary conditions
                fig = ss.show_axial_force(scale=0.55 , factor= 0.5, verbosity=0, show=False, figsize= struct_fig_size)
                plt.title(f'Axial forces (N) (frame number {Frame_number})')
                plt.show()
                
                ## Plot the structure with the loads and boundary conditions
                fig = ss.show_bending_moment(scale=0.55, factor= 500, verbosity=0, show=False, figsize= struct_fig_size)
                plt.title(f'Bending moments (Nm) (frame number {Frame_number})')
                plt.show()
                
                # ## Plot the structure with the loads and boundary conditions
                fig = ss.show_shear_force(scale=0.55, factor= 500, verbosity=0, show=False, figsize= struct_fig_size)
                plt.title(f'Shear forces (N) (frame number {Frame_number})')
                plt.show()
                
                ## Plot the structure with the loads and boundary conditions
                fig = ss.show_displacement(scale=0.55, factor =20 , verbosity=0, show=False, figsize= struct_fig_size)
                plt.title(f'Displacements and deformations (mm) (frame number {Frame_number})')
                plt.show()
                
                # fig_results = ss.show_results(scale=0.55, verbosity=1, show=False, figsize=(fig_size,fig_size))
                # plt.tight_layout()
                # plt.show()
                # plt.close()
                
                # # Create a figure with 6 subplots
                # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                # for ii in range(6):
                #     fig.add_subplot(axis_results[ii])
                # # Adjust the layout to prevent overlap
                # plt.tight_layout()
                
                # # Display the empty subplots
                # plt.show()
        
        
        df_model.loc[:,'Load applied (N)'] = Load_list
        df_model.loc[:,'Delta compression (mm)'] = delta_y_exp_list
        df_model.loc[:,'delta strain'] = delta_strain_exp_list
        df_model.loc[:,'Delta compression model (mm)'] = delta_y_model_list
        df_model.loc[:,'delta strain model'] = strain_model_list
    
        modeling_excel_filename = f'{test_folder}/modeling_data_{meth}_{sample_name}_rate{fps_div}.xlsx'
        # Save the DataFrame to an Excel file
        df_model.to_excel(modeling_excel_filename, index=True)
        
        
        constant = 1 
        constant_2 = 0
        
        
        stress_list_plot = list(df_model['Stress (MPa)'])
        # stress_list_plot.insert(0, 0)
        # strain_model_list.insert(0, 0)
        # delta_strain_exp_list.insert(0, 0)
    
    
    
        # Multiply all values in the list by the constant
        multiplied_strain_list = [x * constant + constant_2 for x in strain_model_list]
        
        plt.figure(figsize=(8,8))
        plt.plot(strain_model_list, stress_list_plot, label='Model', color='black', linestyle='--')
        plt.plot(delta_strain_exp_list, stress_list_plot, label='Experiment', color='black', linestyle='-')
        # plt.title(f'Structure of the currugated board in frame number {Frame_number}')
        plt.xlabel('Strain ')
        plt.ylabel('σ in MPa')
        plt.legend()
        # plt.xlim(0,0.03)
        # plt.ylim(0,0.11)
        plt.grid('on')
    
        plt.show()
        
        print(strain_model_list)
        print(delta_strain_exp_list)    
        
        Force_list_plot = list(df_model['Force (N)'])
        # Force_list_plot.insert(0, 0)
        # delta_y_model_list.insert(0, 0)
        # delta_y_exp_list.insert(0, 0)
        
        
        # Multiply all values in the list by the constant
        multiplied_delta_y_list = [x * constant + constant_2 for x in delta_y_model_list]
    
    
        plt.figure(figsize=(10,10))
        plt.plot(delta_y_model_list, Force_list_plot, label = 'Model', color='black', linestyle='--')
        plt.plot(delta_y_exp_list, Force_list_plot, label = 'Experiment', color='black', linestyle='-')
        # plt.title(f'Structure of the currugated board in frame number {Frame_number}')
        plt.xlabel('Deformation (mm)')
        plt.ylabel('Force (N)')
        plt.legend()
        # plt.xlim(0,0.03)
        # plt.ylim(0,0.11)
        plt.grid('on')
        plt.show()
        
        coeff_correction_list =[]
        for ii in range(len(strain_model_list)):
            coeff_correction_list.append(delta_strain_exp_list[ii]/strain_model_list[ii])
        
        arv_coeff_correction = sum(coeff_correction_list)/len(coeff_correction_list)
        print(arv_coeff_correction)

with open('Graphs_List.pkl', 'wb') as file:
    pickle.dump(Graphs_List, file)