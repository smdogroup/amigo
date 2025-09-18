import numpy as np
import re
from tabulate import tabulate


class InpParser:
    def __init__(self):
        self.elements = {}

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

                if header.startswith("*NODE"):
                    section = "NODE"
                elif header.startswith("*ELEMENT"):
                    section = "ELEMENT"
                    elem_type = self._find_kw(header, "TYPE")
                    elset = self._find_kw(header, "ELSET")
                    if elem_type not in self.elem_conn:
                        self.elem_conn[elset] = {}
                    if elset not in self.elem_conn[elset]:
                        self.elem_conn[elset][elem_type] = {}
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

    def get_nodes(self):
        return np.array([self.X[k] for k in sorted(self.X.keys())])

    def get_names(self):
        names = {}
        for elset in self.elem_conn:
            names[elset] = []
            for elem_type in self.elem_conn[elset]:
                names[elset].append(elem_type)

        return names

    def get_conn(self, elset, elem_type):
        conn = self.elem_conn[elset][elem_type]
        return np.array([conn[k] for k in sorted(conn.keys())], dtype=int)


if __name__ == "__main__":
    # Filename for the mesh
    fname = "plate.inp"

    # Instance of InpParse
    parser = InpParser()
    parser.parse_inp(fname)

    # Get the node locations
    X = parser.get_nodes()

    # Get element connectivity
    conn = parser.get_conn("SURFACE1", "CPS3")

    # Get the boundary condition nodes
    edge1 = parser.get_conn("LINE1", "T3D2")
    edge2 = parser.get_conn("LINE2", "T3D2")
    edge3 = parser.get_conn("LINE3", "T3D2")
    edge4 = parser.get_conn("LINE4", "T3D2")

    # Print out results
    print("Node Coords")
    print(tabulate(X, headers=["x", "y", "z"]))

    print("\nConnectivity")
    print(tabulate(conn, headers=["n1", "n2", "n3"]))

    print("\nBoundary condition edges")
    print("Edge1:", edge1)
    print("Edge2:", edge2)
    print("Edge3:", edge3)
    print("Edge4:", edge4)
