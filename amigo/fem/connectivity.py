from . import basis

import numpy as np
import re


class InpParser:
    def __init__(self):
        self.elements = {}
        self.surfaces = {}

    def _read_file(self, filename):
        with open(filename, "r", errors="ignore") as fp:
            return [line.rstrip("\n") for line in fp]

    def _split_csv_line(self, s):
        # ABAQUS lines are simple CSV-like, no quotes typically
        return [p.strip() for p in s.split(",") if p.strip()]

    def _find_kw(self, header, key):
        m = re.search(rf"\b{re.escape(key)}\s*=\s*([^,\s]+)", header, flags=re.I)
        return m.group(1) if m else None

    def parse_inp(self, filename):
        # Read the entire file in
        lines = self._read_file(filename)

        self.X = {}
        self.elem_conn = {}
        self.node_sets = {}
        self.surfaces = []

        elem_type = None

        index = 0
        section = None
        while index < len(lines):
            raw = lines[index].strip()
            index += 1

            if not raw or raw.startswith("**"):
                continue

            if raw.startswith("*"):
                header = raw.upper()

                if header.startswith("*NODE") and not header.startswith("*NSET"):
                    section = "NODE"
                elif header.startswith("*ELEMENT"):
                    section = "ELEMENT"
                    elem_type = self._find_kw(raw, "TYPE")
                    elset = self._find_kw(raw, "ELSET")
                    if elset not in self.elem_conn:
                        self.elem_conn[elset] = {}
                    if elem_type not in self.elem_conn[elset]:
                        self.elem_conn[elset][elem_type] = {}
                    if "SURFACE" in elset.upper():
                        self.surfaces.append(elset)
                elif header.startswith("*NSET"):
                    section = "NSET"
                    nset_name = self._find_kw(raw, "NSET")
                    if nset_name not in self.node_sets:
                        self.node_sets[nset_name] = []
                else:
                    section = None
                continue

            if section == "NODE":
                parts = self._split_csv_line(raw)
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                self.X[nid - 1] = (x, y, z)

            elif section == "ELEMENT":
                parts = self._split_csv_line(raw)
                eid = int(parts[0])
                conn = [(int(p) - 1) for p in parts[1:]]
                self.elem_conn[elset][elem_type][eid - 1] = conn

            elif section == "NSET":
                parts = self._split_csv_line(raw)
                self.node_sets[nset_name].extend(int(p) - 1 for p in parts)

    def get_nodes(self):
        return np.array([self.X[k] for k in sorted(self.X.keys())])

    def get_domains(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_num_surfaces(self):
        return len(self.surfaces)

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset][elem_type]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)

    def get_nodes_in_domain(self, elset):
        if elset in self.node_sets:
            return np.array(list(dict.fromkeys(self.node_sets[elset])))

        conn = []
        for elem_type in self.elem_conn[elset]:
            conn.extend(self.get_conn(elset, elem_type).flatten())

        # Turn into a single unique list of nodes preserving GMSH ordering
        node_list = np.array(list(dict.fromkeys(conn)))

        return node_list

    def get_basis(self, space, etype, kind="input"):
        basis_list = []

        for sp in ["H1", "const"]:
            names = space.get_names(sp)

            if len(names) == 0:
                continue

            basis_list.append(self._get_basis(etype, sp, names, kind))

        return basis.BasisCollection(basis_list)

    def _get_basis(self, etype, space, names=[], kind="input"):
        if etype == "CPS3":
            if space == "H1":
                return basis.TriangleLagrangeBasis(1, names, kind=kind)
            elif space == "const":
                return basis.ConstantBasis(names=names, kind=kind)
        elif etype == "CPS4":
            if space == "H1":
                return basis.QuadLagrangeBasis(1, names, kind=kind)
            elif space == "const":
                return basis.ConstantBasis(names=names, kind=kind)
        elif etype == "CPS6":
            if space == "H1":
                return basis.TriangleLagrangeBasis(2, names, kind=kind)
        elif etype == "M3D9":
            if space == "H1":
                return basis.QuadLagrangeBasis(2, names, kind=kind)
        elif etype == "T3D2":
            if space == "H1":
                return basis.LagrangeBasis1D(1, names, kind=kind)

        raise NotImplementedError(
            f"Basis for element {etype} with space {space} not implemented"
        )

    def get_quadrature(self, etype):
        if etype == "CPS3":
            return basis.TriangleQuadrature(2)
        elif etype == "CPS4":
            return basis.QuadQuadrature(2)
        elif etype == "CPS6":
            return basis.TriangleQuadrature(4)
        elif etype == "M3D9":
            return basis.QuadQuadrature(3)
        elif etype == "T3D2":
            return basis.LineQuadrature(2)

        raise NotImplementedError(f"Quadrature for element {etype} not implemented")

    def get_conn_edges(self, elset, elem_type):
        """
        Build a unique-edge connectivity dictionary for the given elset/element type.

        Returns
        -------
        edge_dict : dict
            Keys are integer edge tags (0, 1, 2, …), one per unique global edge.
            Values are tuples of global node indices that make up that edge.
            The first two entries are always the corner nodes in ascending order
            (lowest index first).  For higher-order elements a third entry
            carries the mid-side node, preserving its original orientation so
            that the mid-side node is correctly associated with its edge even
            after the corners are sorted.

        Example (linear triangle mesh with 4 nodes, 2 elements)
        -------------------------------------------------------
            edge_dict = {
                0: (0, 1),
                1: (1, 2),
                2: (0, 2),
                3: (2, 3),
                4: (1, 3),
            }
        """
        # Local edge definitions per element type.
        # Each entry is a list of (local_node_i, local_node_j) pairs that form
        # the edges of that element. Corner nodes only — mid-side nodes are
        # included as the third entry where present so the full edge is
        # (corner_a, corner_b, optional_midside).
        _EDGE_LOCAL = {
            # Linear triangle: 3 edges
            "CPS3": [(0, 1), (1, 2), (2, 0)],
            # Linear quad: 4 edges
            "CPS4": [(0, 1), (1, 2), (2, 3), (3, 0)],
            # Quadratic triangle: 3 edges, each with a mid-side node
            #   0-3-1 / 1-4-2 / 2-5-0
            "CPS6": [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
            # 9-node quad (serendipity/Lagrange):
            #   0-4-1 / 1-5-2 / 2-6-3 / 3-7-0
            "M3D9": [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
        }

        if elem_type not in _EDGE_LOCAL:
            raise NotImplementedError(
                f"Edge connectivity not defined for element type '{elem_type}'. "
                f"Supported types: {list(_EDGE_LOCAL.keys())}"
            )

        H1_tri_conn = self.get_conn(elset, elem_type)  # shape (nelems, nodes_per_elem)
        local_edges = _EDGE_LOCAL[elem_type]

        seen = {}  # canonical_corner_pair -> edge_tag
        edge_dict = {}  # edge_tag -> full node tuple (corner0, corner1[, midside])
        tag = 0

        for elem_nodes in H1_tri_conn:
            for local_edge in local_edges:
                # Corner nodes that define uniqueness
                n0 = int(elem_nodes[local_edge[0]])
                n1 = int(elem_nodes[local_edge[1]])

                # Canonical key: lowest node index first
                key = (min(n0, n1), max(n0, n1))

                if key not in seen:
                    seen[key] = tag

                    # Build the full tuple: sorted corners + optional mid-side
                    if len(local_edge) == 3:
                        mid = int(elem_nodes[local_edge[2]])
                        # Mid-side node follows the corner ordering
                        full = (
                            (key[0], key[1], mid) if n0 < n1 else (key[0], key[1], mid)
                        )
                        edge_dict[tag] = full
                    else:
                        edge_dict[tag] = key

                    tag += 1

        return edge_dict
