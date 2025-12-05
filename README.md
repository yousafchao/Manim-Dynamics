# Manim-Dynamics

A Manim-powered library for animating attractors and phase portraits of classic dynamical systems (e.g., Lorenz, Rössler), built as a learning-first toolkit with applications in neuroscience and anesthesia research.

## Requirements
- Python 3.12
- Manim Community v0.19+

## Run (4K / 60 FPS)
```bash
conda activate manim
PYTHONPATH=. manim -p --disable_caching -qh --resolution 3840,2160 --fps 60 mylib/scenes/LorenzWakeScene.py LorenzWakeScene
PYTHONPATH=. manim -p --disable_caching -qh --resolution 3840,2160 --fps 60 mylib/scenes/LorenzAnesthesiaScene.py LorenzAnesthesiaScene

