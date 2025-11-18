---
sidebar_position: 5
---

# ExternalComponent API

Interface for integrating external models (e.g., OpenMDAO) with Amigo.

## Overview

`ExternalComponent` allows you to wrap external analysis codes and integrate them into Amigo models while maintaining automatic differentiation support.

## Basic Usage

```python
from amigo import ExternalComponent

class MyExternalWrapper(ExternalComponent):
    def __init__(self):
        super().__init__()
        # Define interface
        self.add_input("x", value=0.0)
        self.add_output("y")
    
    def compute_external(self, inputs):
        # Call external code
        result = external_function(inputs["x"])
        return {"y": result}
    
    def compute_derivatives(self, inputs):
        # Provide derivatives (if available)
        return {"dy/dx": derivative_function(inputs["x"])}
```

Complete documentation for ExternalComponent will be provided in future releases.

