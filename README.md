# Autonomous 3D Robot Fleet Simulation
- This project simulates an autonomous fleet of robots navigating a procedurally generated city grid, detecting hazards, and performing roles like scouting, carrying, and blocking. 
- The simulation runs entirely in the browser using Pyodide (Python in WebAssembly) for logic and Three.js for 3D rendering. 
- No server-side computation or installations are required; everything is client-side!

## Features
- Procedural City Generation: A 10x10 grid city with buildings and roads, generated randomly each reset.
- Multi-Agent System: Up to 7 robots (wheeled, legged, hybrid) with AI-driven roles (scout, carrier, blocker, idle).
- Hazard Detection: Randomly placed fires and blockages that robots can detect within a radius.
- Collision Handling: Robots avoid buildings and each other with physics-based resolution.
- Neural Network Policies: Simple random neural nets for role selection and actions (no training, just demo).
- Real-Time 3D Visualization: Interactive camera controls, animations, lighting, and fog effects.
- Debug Tools: Robot vision overlay, stats panel, legend, and WebGPU badge (if supported).
- Controls: Pause/resume, reset, speed adjustment (1x/2x/4x), and vision debug toggle.
- State Persistence: Simulation state is saved to output/simulation_state.json after each step/reset for external analysis.
- Stats Tracking: Timesteps, detected hazards, average rewards, role distribution, collisions, and FPS.

## Technologies Used
- Frontend: HTML5, CSS3, JavaScript
- 3D Rendering: Three.js (r128) with OrbitControls
- Python Runtime: Pyodide (v0.24.1) for browser-based Python execution
- Libraries (via Pyodide): NumPy for math/computations
- Browser Features: WebGL (fallback; WebGPU noted but requires Three.js upgrade for full support)

## Requirements
- Modern web browser (Chrome, Firefox, Edge recommended; tested on Chrome 120+).
- Local HTTP server for serving files (due to CORS restrictions with file:// protocol when fetching scripts/packages).
- No Python installation needed—everything runs in the browser!

## Setup and Running
1. Clone/Download the Repository:
- `git clone <repo-url>`
- `cd autonomous-robot-fleet`

2. Serve the Files Locally:
- If you have Python: `python -m http.server 8000`
- If you have Node.js: Install `http-server` via `npm install -g http-server` and `run http-server.`
- Open your browser to `http://localhost:8000/robot_fleet_visual.html`

3. Run the Simulation:
- Open `robot_fleet_visual.html` in your browser (via the local server).
- The page will load Pyodide, fetch NumPy, and initialize the simulation (may take 10-30 seconds on first load).
- Once loaded, the 3D scene appears with controls at the bottom.

4. Output Files:
- Simulation state is automatically saved to `./output/simulation_state.json` after each step or reset.
- This JSON includes city layout, hazards, agents' positions/roles, stats, etc.—useful for debugging or integration.

## Controls
- Orbit Camera: Drag to rotate, scroll to zoom, right-click drag to pan.
- Pause/Play: Toggle simulation running.
- Reset: Regenerate city, hazards, and robots.
- Speed: Cycle between 1x, 2x, and 4x simulation speed.
- Vision: Toggle robot vision debug (shows a top-down radar view for a selected robot; click again to cycle robots).

## Project Structure
- `robot_fleet_visual.html`: Main HTML file with embedded JS for rendering and simulation control.
- `robot_fleet_logic.py`: Python logic for city generation, agents, hazards, neural nets, and simulation stepping.
- `output/:` Auto-generated folder for `simulation_state.json`.

## Contributing
- Feel free to fork, improve, or submit issues/PRs!
- Ideas: Add real RL training, more robot types, or multiplayer controls.

## License
- MIT License: free to use/modify with attribution.
- Created by [Melisa Sever]–November 2025.
