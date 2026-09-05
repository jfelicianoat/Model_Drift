"""Model_Drift — detección de deriva de comportamiento en LLMs.

Mide si los prompts de un producto siguen comportándose igual cuando cambia el
modelo o su configuración de ejecución. No juzga calidad absoluta: detecta
cambio, y solo asigna dirección (mejora/regresión) cuando tiene evidencia
evaluativa para hacerlo.

Documento de referencia: `docs/Model_Drift_Plan.md` (revisión 2).
"""
from __future__ import annotations

__version__ = "0.2.0"
