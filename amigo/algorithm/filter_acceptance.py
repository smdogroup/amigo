"""Two-dimensional filter used by the filter line search.

Stores (phi, theta) pairs with a built-in safety margin.  A trial
point is acceptable when no stored entry dominates it in both the
barrier objective phi and the constraint violation theta.
"""


class Filter:
    """Filter for the line search.

    Stores (phi, theta) pairs with margins baked in at add-time.
    Acceptance is a strict dominance check: a trial point is acceptable
    if it is NOT dominated by any filter entry.

    The margins match Eq. 18:
      phi_entry   = phi_ref - gamma_phi * theta_ref
      theta_entry = (1 - gamma_theta) * theta_ref
    """

    def __init__(self, gamma_phi=1e-8, gamma_theta=1e-5):
        self.entries = []
        self.gamma_phi = gamma_phi
        self.gamma_theta = gamma_theta

    def is_acceptable(self, phi_trial, theta_trial):
        """True if trial is not dominated by any filter entry.

        A trial is acceptable to an entry if at least one coordinate
        is <= the entry. Rejection requires STRICT > in ALL coordinates.
        """
        for phi_f, theta_f in self.entries:
            if phi_trial > phi_f and theta_trial > theta_f:
                return False
        return True

    def add(self, phi, theta):
        """Add entry with margins (Eq. 18), remove dominated."""
        phi_entry = phi - self.gamma_phi * theta
        theta_entry = (1.0 - self.gamma_theta) * theta
        self.entries = [
            (p, t) for p, t in self.entries if not (p >= phi_entry and t >= theta_entry)
        ]
        self.entries.append((phi_entry, theta_entry))

    def clear(self):
        """Remove all filter entries."""
        self.entries.clear()

    def __len__(self):
        return len(self.entries)
