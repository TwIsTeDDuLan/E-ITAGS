#!/usr/bin/env python3
"""Demonstration: TraitEngine Reasoning & WarehouseVisualizer Integration in E-ITAGS.

This script demonstrates:
1. TraitEngine:
   - Domain-specific trait aggregation (Sum, Max, Min physics rules)
   - Realizability analysis: discrete physical trait set vs continuous solver delusion
   - Trait satisfaction scoring: hard constraint enforcement vs soft matching
2. WarehouseVisualizer:
   - 2D floor map with realistic warehouse zones
   - Spatial task allocation & coalition dispatch arrows
   - Trait satisfaction heatmap matrix
   - Physical realizability space comparison
   - Multi-agent trait synergy breakdown for collaborative tasks
   - Comprehensive 4-panel publication-ready dashboard
3. Interactive HTML Dashboard export for live browser exploration
"""
from __future__ import annotations
from pathlib import Path
import json
import yaml

from src.models.robot import Robot, RobotType, RobotTraits, TRAIT_NAMES
from src.models.task import Task, TaskRequirements
from src.models.coalition import Coalition
from src.models.warehouse import WarehouseGrid
from src.core.trait_engine import TraitEngine
from src.utils.display import TerminalDisplay
from src.visualization.warehouse_viz import WarehouseVisualizer


def load_warehouse_scenario(config_path: Path):
    """Load warehouse scenario configuration from YAML."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    robots: list[Robot] = []
    for r_data in data["robots"]:
        r_type = r_data["type"]
        if r_type == "forklift":
            r = Robot.create_forklift(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        elif r_type == "drone":
            r = Robot.create_drone(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        elif r_type == "iw_hub":
            r = Robot.create_iw_hub(r_data["id"], r_data["name"], r_data.get("initial_zone", "staging"))
        else:
            raise ValueError(f"Unknown robot type: {r_type}")
        robots.append(r)

    tasks: list[Task] = []
    for t_data in data["tasks"]:
        req_data = t_data["requirements"]
        req = TaskRequirements(
            min_payload_kg=float(req_data.get("min_payload_kg", 0.0)),
            min_has_lidar_2d=float(req_data.get("min_has_lidar_2d", 0.0)),
            min_has_camera_rgb=float(req_data.get("min_has_camera_rgb", 0.0)),
            min_vertical_lift=float(req_data.get("min_vertical_lift", 0.0)),
            min_narrow_space_access=float(req_data.get("min_narrow_space_access", 0.0)),
        )
        task = Task(
            id=t_data["id"],
            name=t_data["name"],
            zone=t_data["zone"],
            duration_seconds=float(t_data["duration_seconds"]),
            requirements=req,
            description=t_data.get("description", ""),
        )
        tasks.append(task)

    return data["scenario_name"], robots, tasks


def export_interactive_html(
    warehouse: WarehouseGrid,
    robots: list[Robot],
    tasks: list[Task],
    coalitions: list[Coalition],
    allocations: dict[str, Coalition],
    html_path: Path,
):
    """Generates a standalone, interactive HTML+CSS+JS dashboard."""
    # Prepare serializable scenario data
    scenario_data = {
        "warehouse": {
            "width": warehouse.width,
            "height": warehouse.height,
            "zones": {
                name: {
                    "description": z.description,
                    "center": {"x": z.center.x, "y": z.center.y},
                    "narrow_access_only": z.narrow_access_only,
                    "cells": [{"x": c.x, "y": c.y} for c in z.cells],
                }
                for name, z in warehouse.zones.items()
            },
        },
        "robots": [
            {
                "id": r.id,
                "name": r.name,
                "type": r.robot_type.value,
                "initial_zone": r.initial_zone,
                "traits": r.traits.to_dict(),
            }
            for r in robots
        ],
        "tasks": [
            {
                "id": t.id,
                "name": t.name,
                "zone": t.zone,
                "duration_seconds": t.duration_seconds,
                "description": t.description,
                "requirements": t.requirements.to_dict(),
            }
            for t in tasks
        ],
        "coalitions": [
            {
                "name": str(c),
                "robot_ids": list(c.robot_ids),
                "size": c.size,
                "traits": c.aggregate_traits.to_dict(),
            }
            for c in coalitions
        ],
        "allocations": {t_id: str(c) for t_id, c in allocations.items()},
    }

    # Precompute satisfaction matrix
    satisfaction_matrix = {}
    for c in coalitions:
        satisfaction_matrix[str(c)] = {}
        for t in tasks:
            score, breakdown = TraitEngine.compute_trait_satisfaction(c, t)
            satisfaction_matrix[str(c)][t.id] = {
                "score": round(score, 2),
                "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
            }
    scenario_data["satisfaction_matrix"] = satisfaction_matrix

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>E-ITAGS: TraitEngine & Warehouse Visualization</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-primary: #0F172A;
      --bg-secondary: #1E293B;
      --bg-card: rgba(30, 41, 59, 0.7);
      --border-color: rgba(148, 163, 184, 0.15);
      --text-main: #F8FAFC;
      --text-muted: #94A3B8;
      --accent-blue: #38BDF8;
      --accent-purple: #A855F7;
      --accent-emerald: #10B981;
      --accent-amber: #F59E0B;
      --accent-rose: #F43F5E;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 10% 20%, #0f172a 0%, #030712 100%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-color);
    }}
    h1 {{
      font-size: 26px;
      font-weight: 800;
      background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34D399;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
    }}
    .pulse-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10B981;
      box-shadow: 0 0 10px #10B981;
      animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; transform: scale(1); }}
      50% {{ opacity: 0.4; transform: scale(0.85); }}
    }}
    .main-grid {{
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 24px;
    }}
    @media (max-width: 1200px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}
    .panel {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
      position: relative;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }}
    .panel-title {{
      font-size: 16px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #map-container {{
      position: relative;
      background: #020617;
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 12px;
      overflow: hidden;
      aspect-ratio: 30 / 20;
      width: 100%;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .zone-rect {{
      transition: all 0.25s ease;
      cursor: pointer;
    }}
    .zone-rect:hover {{
      fill-opacity: 0.85;
      stroke-width: 2.5;
    }}
    .task-node, .robot-node {{
      cursor: pointer;
      transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}
    .task-node:hover, .robot-node:hover {{
      filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.8));
    }}
    .dispatch-ray {{
      stroke-dasharray: 4, 4;
      animation: dash 1.5s linear infinite;
    }}
    @keyframes dash {{
      to {{ stroke-dashoffset: -16; }}
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
      background: rgba(15, 23, 42, 0.6);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border-color);
    }}
    .tab-btn {{
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--bg-secondary);
      color: var(--text-main);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}
    .task-card, .coalition-card {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 12px 14px;
      margin-bottom: 10px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .task-card:hover, .coalition-card:hover {{
      border-color: var(--accent-blue);
      transform: translateX(4px);
    }}
    .task-card.active {{
      border-color: var(--accent-blue);
      background: rgba(56, 189, 248, 0.08);
    }}
    .card-title {{
      font-size: 14px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 6px;
      font-weight: 600;
    }}
    .badge-pass {{
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .badge-partial {{
      background: rgba(245, 158, 11, 0.15);
      color: #FBBF24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .badge-fail {{
      background: rgba(244, 63, 94, 0.15);
      color: #FB7185;
      border: 1px solid rgba(244, 63, 94, 0.3);
    }}
    .detail-section {{
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px solid var(--border-color);
      font-size: 13px;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
      margin-top: 8px;
    }}
    .detail-item {{
      background: rgba(15, 23, 42, 0.5);
      padding: 6px 10px;
      border-radius: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11.5px;
      border: 1px solid rgba(148, 163, 184, 0.08);
    }}
    .detail-item span {{
      color: var(--text-muted);
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      margin-bottom: 2px;
    }}
    .realizability-box {{
      background: rgba(15, 23, 42, 0.6);
      border-radius: 12px;
      padding: 16px;
      margin-top: 16px;
      border: 1px solid var(--border-color);
    }}
    .input-row {{
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }}
    input[type="number"] {{
      flex: 1;
      background: #020617;
      border: 1px solid rgba(148, 163, 184, 0.2);
      color: var(--text-main);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
    }}
    input[type="number"]:focus {{
      outline: none;
      border-color: var(--accent-blue);
    }}
    .btn {{
      background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
      border: none;
      color: white;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn:hover {{
      opacity: 0.9;
      transform: translateY(-1px);
    }}
    #query-result {{
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 12.5px;
      display: none;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>E-ITAGS Interactive Trait Demonstration</h1>
      <p class="subtitle">Realizability-Aware Multi-Robot Trait Allocation & Warehouse Floor Map</p>
    </div>
    <div class="status-pill">
      <div class="pulse-dot"></div>
      <span>TraitEngine v0.1.0 Online</span>
    </div>
  </header>

  <div class="main-grid">
    <!-- Left: 2D Spatial Visualization -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <span>📍</span> 2D Warehouse Spatial Map & Coalition Allocations
        </div>
        <div style="font-size: 12px; color: var(--text-muted);">
          Click any task or robot on the map to inspect
        </div>
      </div>
      <div id="map-container">
        <svg id="warehouse-svg" viewBox="-1 -1 32 22"></svg>
      </div>

      <!-- Realizability Interactive Query -->
      <div class="realizability-box">
        <div style="font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px;">
          <span>⚡</span> TraitEngine Realizability Checker (Core Innovation)
        </div>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
          Continuous solvers assume any payload can be allocated. Physically, fleet traits are discrete.
        </p>
        <div class="input-row">
          <input type="number" id="payload-input" value="1500" step="100" min="0" max="4000" placeholder="Payload kg (e.g. 100, 500, 1000, 1500, 2000)">
          <button class="btn" id="check-btn" onclick="checkRealizability()">Check Realizability</button>
        </div>
        <label style="display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-muted); margin-top: 8px; cursor: pointer;">
          <input type="checkbox" id="surplus-checkbox" style="accent-color: #38BDF8;">
          Allow surplus capacity (Q(C) &ge; target threshold)
        </label>
        <div id="query-result"></div>
      </div>
    </div>

    <!-- Right: Trait Satisfaction & Coalition Diagnostics -->
    <div class="panel">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tasks')">Warehouse Tasks (5)</button>
        <button class="tab-btn" onclick="switchTab('coalitions')">Formable Coalitions (7)</button>
        <button class="tab-btn" onclick="switchTab('robots')">Physical Fleet (3)</button>
      </div>

      <div id="tab-content" style="max-height: 600px; overflow-y: auto; padding-right: 4px;">
        <!-- Dynamic cards inserted via JS -->
      </div>

      <div id="inspector-panel" class="detail-section" style="display: none;">
        <!-- Inspector detail view -->
      </div>
    </div>
  </div>

  <script>
    const data = {json.dumps(scenario_data)};

    let activeTaskId = "T4"; // Default to collaborative task
    let currentTab = "tasks";

    // Setup Map
    const svg = document.getElementById("warehouse-svg");
    const zoneColors = {{
      receiving: {{ fill: "#0369A1", stroke: "#38BDF8" }},
      sorting: {{ fill: "#B45309", stroke: "#FBBF24" }},
      staging: {{ fill: "#7E22CE", stroke: "#C084FC" }},
      narrow_aisle: {{ fill: "#BE123C", stroke: "#FB7185" }},
      shipping: {{ fill: "#047857", stroke: "#34D399" }}
    }};

    function renderMap() {{
      svg.innerHTML = `
        <defs>
          <pattern id="grid" width="1" height="1" patternUnits="userSpaceOnUse">
            <path d="M 1 0 L 0 0 0 1" fill="none" stroke="rgba(148, 163, 184, 0.08)" stroke-width="0.05"/>
          </pattern>
        </defs>
        <rect x="-1" y="-1" width="32" height="22" fill="#020617"/>
        <rect x="0" y="0" width="30" height="20" fill="url(#grid)"/>
      `;

      // Draw Zones (invert Y for SVG rendering: y_svg = 20 - y)
      for (const [zName, zone] of Object.entries(data.warehouse.zones)) {{
        if (zone.cells.length > 0) {{
          const xs = zone.cells.map(c => c.x);
          const ys = zone.cells.map(c => c.y);
          const minX = Math.min(...xs);
          const maxX = Math.max(...xs) + 1;
          const minY = Math.min(...ys);
          const maxY = Math.max(...ys) + 1;

          const color = zoneColors[zName] || {{ fill: "#334155", stroke: "#64748B" }};
          const w = maxX - minX;
          const h = maxY - minY;
          const svgY = 20 - maxY;

          const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("x", minX);
          rect.setAttribute("y", svgY);
          rect.setAttribute("width", w);
          rect.setAttribute("height", h);
          rect.setAttribute("fill", color.fill);
          rect.setAttribute("fill-opacity", "0.2");
          rect.setAttribute("stroke", color.stroke);
          rect.setAttribute("stroke-width", zone.narrow_access_only ? "0.3" : "0.2");
          if (zone.narrow_access_only) rect.setAttribute("stroke-dasharray", "0.6, 0.3");
          rect.setAttribute("rx", "0.4");
          rect.classList.add("zone-rect");
          svg.appendChild(rect);

          // Zone text
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", minX + w / 2);
          text.setAttribute("y", svgY + 1.2);
          text.setAttribute("fill", color.stroke);
          text.setAttribute("font-size", "0.75");
          text.setAttribute("font-weight", "bold");
          text.setAttribute("text-anchor", "middle");
          text.textContent = zName.replace('_', ' ').toUpperCase();
          svg.appendChild(text);
        }}
      }}

      // Robot positions in staging
      const robotCoords = {{}};
      data.robots.forEach((r, idx) => {{
        const cx = 5 - 2.5 + (idx * 2.5);
        const cy = 20 - 7 - 0.5;
        robotCoords[r.id] = {{ x: cx, y: cy }};

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.classList.add("robot-node");
        g.onclick = () => selectRobot(r.id);

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", "0.75");
        circle.setAttribute("fill", r.type === "forklift" ? "#2563EB" : (r.type === "drone" ? "#9333EA" : "#059669"));
        circle.setAttribute("stroke", "#FFFFFF");
        circle.setAttribute("stroke-width", "0.15");
        g.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", cx);
        label.setAttribute("y", cy + 0.3);
        label.setAttribute("fill", "#FFFFFF");
        label.setAttribute("font-size", "0.55");
        label.setAttribute("font-weight", "bold");
        label.setAttribute("text-anchor", "middle");
        label.textContent = r.id.replace("R_", "");
        g.appendChild(label);

        svg.appendChild(g);
      }});

      // Draw Task points
      const taskCoords = {{}};
      data.tasks.forEach((t, i) => {{
        const z = data.warehouse.zones[t.zone];
        const tx = z.center.x + ((i % 2 - 0.5) * 1.6);
        const ty = 20 - (z.center.y + (Math.floor(i / 2) - 0.5) * 1.6);
        taskCoords[t.id] = {{ x: tx, y: ty }};

        // If this task is active, draw rays from its assigned coalition
        if (t.id === activeTaskId) {{
          const assignedCoalitionName = data.allocations[t.id];
          const coalitionObj = data.coalitions.find(c => c.name === assignedCoalitionName);
          if (coalitionObj) {{
            coalitionObj.robot_ids.forEach(rId => {{
              const rPos = robotCoords[rId];
              if (rPos) {{
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", rPos.x);
                line.setAttribute("y1", rPos.y);
                line.setAttribute("x2", tx);
                line.setAttribute("y2", ty);
                line.setAttribute("stroke", "#38BDF8");
                line.setAttribute("stroke-width", "0.22");
                line.classList.add("dispatch-ray");
                svg.appendChild(line);
              }}
            }});
          }}
        }}

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.classList.add("task-node");
        g.onclick = () => selectTask(t.id);

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", tx - 0.7);
        rect.setAttribute("y", ty - 0.7);
        rect.setAttribute("width", "1.4");
        rect.setAttribute("height", "1.4");
        rect.setAttribute("rx", "0.25");
        rect.setAttribute("fill", t.id === activeTaskId ? "#38BDF8" : "#E11D48");
        rect.setAttribute("stroke", "#FFFFFF");
        rect.setAttribute("stroke-width", t.id === activeTaskId ? "0.25" : "0.12");
        g.appendChild(rect);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", tx);
        label.setAttribute("y", ty + 0.3);
        label.setAttribute("fill", "#FFFFFF");
        label.setAttribute("font-size", "0.65");
        label.setAttribute("font-weight", "bold");
        label.setAttribute("text-anchor", "middle");
        label.textContent = t.id;
        g.appendChild(label);

        svg.appendChild(g);
      }});
    }}

    function switchTab(tab) {{
      currentTab = tab;
      document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
      event.target.classList.add("active");
      renderTabContent();
    }}

    function renderTabContent() {{
      const container = document.getElementById("tab-content");
      container.innerHTML = "";

      if (currentTab === "tasks") {{
        data.tasks.forEach(task => {{
          const card = document.createElement("div");
          card.className = `task-card ${{task.id === activeTaskId ? 'active' : ''}}`;
          card.onclick = () => selectTask(task.id);

          const assignedCoalition = data.allocations[task.id];
          const sat = data.satisfaction_matrix[assignedCoalition][task.id];
          const badgeClass = sat.score === 1.0 ? 'badge-pass' : (sat.score > 0 ? 'badge-partial' : 'badge-fail');

          card.innerHTML = `
            <div class="card-title">
              <span><strong>${{task.id}}</strong>: ${{task.name}}</span>
              <span class="badge ${{badgeClass}}">S = ${{sat.score.toFixed(2)}}</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Zone: <em>${{task.zone}}</em> | ${{task.description}}</p>
            <div style="font-size: 11.5px; color: #38BDF8; margin-top: 6px; font-weight: 600;">
              Optimal Coalition: ${{assignedCoalition}}
            </div>
          `;
          container.appendChild(card);
        }});
      }} else if (currentTab === "coalitions") {{
        data.coalitions.forEach(c => {{
          const card = document.createElement("div");
          card.className = "coalition-card";
          card.onclick = () => selectCoalition(c.name);

          card.innerHTML = `
            <div class="card-title">
              <span>${{c.name}}</span>
              <span class="badge" style="background: rgba(148,163,184,0.1); color: #CBD5E1;">Size: ${{c.size}}</span>
            </div>
            <div style="display: flex; gap: 8px; font-family: 'JetBrains Mono'; font-size: 11px; margin-top: 6px; color: #94A3B8;">
              <span>⚖️ ${{c.traits.payload_kg}}kg</span>
              <span>👁️ LiDAR: ${{c.traits.has_lidar_2d ? 'YES' : 'NO'}}</span>
              <span>📷 RGB: ${{c.traits.has_camera_rgb}}</span>
              <span>🏗️ Mast: ${{c.traits.vertical_lift ? 'YES' : 'NO'}}</span>
            </div>
          `;
          container.appendChild(card);
        }});
      }} else if (currentTab === "robots") {{
        data.robots.forEach(r => {{
          const card = document.createElement("div");
          card.className = "coalition-card";
          card.onclick = () => selectRobot(r.id);

          card.innerHTML = `
            <div class="card-title">
              <span><strong>${{r.id}}</strong>: ${{r.name}}</span>
              <span class="badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8;">${{r.type.toUpperCase()}}</span>
            </div>
            <div class="detail-grid">
              <div class="detail-item"><span>Payload</span>${{r.traits.payload_kg}} kg</div>
              <div class="detail-item"><span>Speed</span>${{r.traits.speed_mps}} m/s</div>
              <div class="detail-item"><span>2D LiDAR</span>${{r.traits.has_lidar_2d ? '1.0 (YES)' : '0.0 (NO)'}}</div>
              <div class="detail-item"><span>RGB Camera</span>${{r.traits.has_camera_rgb}}</div>
              <div class="detail-item"><span>Vertical Lift</span>${{r.traits.vertical_lift ? '1.0 (YES)' : '0.0 (NO)'}}</div>
              <div class="detail-item"><span>Narrow Access</span>${{r.traits.narrow_space_access ? '1.0 (YES)' : '0.0 (NO)'}}</div>
            </div>
          `;
          container.appendChild(card);
        }});
      }}
    }}

    function selectTask(taskId) {{
      activeTaskId = taskId;
      renderMap();
      renderTabContent();

      const task = data.tasks.find(t => t.id === taskId);
      const inspector = document.getElementById("inspector-panel");
      inspector.style.display = "block";

      let coalitionScoringHtml = "";
      data.coalitions.forEach(c => {{
        const res = data.satisfaction_matrix[c.name][taskId];
        const badge = res.score === 1.0 ? 'badge-pass' : (res.score > 0 ? 'badge-partial' : 'badge-fail');
        coalitionScoringHtml += `
          <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(148,163,184,0.06);">
            <span>${{c.name}}</span>
            <span class="badge ${{badge}}">${{res.score.toFixed(2)}}</span>
          </div>
        `;
      }});

      inspector.innerHTML = `
        <div style="font-weight: 700; font-size: 14px; color: #38BDF8; margin-bottom: 4px;">
          TraitEngine Diagnostic: Task ${{task.id}} (${{task.name}})
        </div>
        <p style="color: var(--text-muted); font-size: 12px;">Requirements: ${{task.description}}</p>
        <div class="detail-grid" style="margin-top: 10px;">
          <div class="detail-item"><span>Min Payload</span>${{task.requirements.min_payload_kg}} kg</div>
          <div class="detail-item"><span>LiDAR 2D</span>${{task.requirements.min_has_lidar_2d ? 'REQUIRED' : 'NONE'}}</div>
          <div class="detail-item"><span>RGB Camera</span>${{task.requirements.min_has_camera_rgb ? task.requirements.min_has_camera_rgb : 'NONE'}}</div>
          <div class="detail-item"><span>Mast Lift</span>${{task.requirements.min_vertical_lift ? 'REQUIRED' : 'NONE'}}</div>
          <div class="detail-item"><span>Narrow Space</span>${{task.requirements.min_narrow_space_access ? 'RESTRICTED' : 'STANDARD'}}</div>
        </div>
        <div style="margin-top: 14px; font-weight: 600; font-size: 12.5px;">All Coalition Scores S(C_k, ${{task.id}}):</div>
        <div style="margin-top: 6px; font-size: 12px; max-height: 140px; overflow-y: auto;">
          ${{coalitionScoringHtml}}
        </div>
      `;
    }}

    function selectCoalition(name) {{
      const c = data.coalitions.find(item => item.name === name);
      const inspector = document.getElementById("inspector-panel");
      inspector.style.display = "block";
      inspector.innerHTML = `
        <div style="font-weight: 700; font-size: 14px; color: #C084FC; margin-bottom: 4px;">
          Coalition Profile: ${{c.name}}
        </div>
        <p style="color: var(--text-muted); font-size: 12px;">Aggregated according to TraitEngine physical rules:</p>
        <div class="detail-grid" style="margin-top: 8px;">
          <div class="detail-item"><span>Total Payload</span>${{c.traits.payload_kg}} kg (SUM)</div>
          <div class="detail-item"><span>LiDAR 2D</span>${{c.traits.has_lidar_2d ? '1.0' : '0.0'}} (MAX)</div>
          <div class="detail-item"><span>RGB Camera</span>${{c.traits.has_camera_rgb}} (MAX)</div>
          <div class="detail-item"><span>Mast Lift</span>${{c.traits.vertical_lift ? '1.0' : '0.0'}} (MAX)</div>
          <div class="detail-item"><span>Narrow Access</span>${{c.traits.narrow_space_access ? '1.0' : '0.0'}} (MIN)</div>
          <div class="detail-item"><span>Team Speed</span>${{c.traits.speed_mps}} m/s (MIN)</div>
        </div>
      `;
    }}

    function selectRobot(id) {{
      const r = data.robots.find(item => item.id === id);
      const inspector = document.getElementById("inspector-panel");
      inspector.style.display = "block";
      inspector.innerHTML = `
        <div style="font-weight: 700; font-size: 14px; color: #34D399; margin-bottom: 4px;">
          Robot Profile: ${{r.name}} (${{r.id}})
        </div>
        <p style="color: var(--text-muted); font-size: 12px;">Archetype: ${{r.type.toUpperCase()}}</p>
        <div class="detail-grid" style="margin-top: 8px;">
          <div class="detail-item"><span>Payload</span>${{r.traits.payload_kg}} kg</div>
          <div class="detail-item"><span>Speed</span>${{r.traits.speed_mps}} m/s</div>
          <div class="detail-item"><span>LiDAR 2D</span>${{r.traits.has_lidar_2d ? 'YES' : 'NO'}}</div>
          <div class="detail-item"><span>RGB Camera</span>${{r.traits.has_camera_rgb}}</div>
          <div class="detail-item"><span>Vertical Lift</span>${{r.traits.vertical_lift ? 'YES' : 'NO'}}</div>
          <div class="detail-item"><span>Narrow Access</span>${{r.traits.narrow_space_access ? 'YES' : 'NO'}}</div>
        </div>
      `;
    }}

    function checkRealizability() {{
      const val = parseFloat(document.getElementById("payload-input").value);
      const allowSurplus = document.getElementById("surplus-checkbox").checked;
      const resBox = document.getElementById("query-result");
      resBox.style.display = "block";

      const realizablePayloads = Array.from(new Set(data.coalitions.map(c => c.traits.payload_kg))).sort((a,b)=>a-b);
      let match;
      if (allowSurplus) {{
        match = data.coalitions.find(c => c.traits.payload_kg >= val);
      }} else {{
        match = data.coalitions.find(c => Math.abs(c.traits.payload_kg - val) < 1e-3);
      }}

      if (match) {{
        resBox.style.background = "rgba(16, 185, 129, 0.15)";
        resBox.style.border = "1px solid rgba(16, 185, 129, 0.4)";
        resBox.style.color = "#34D399";
        const modeText = allowSurplus ? `Surplus Coverage (capacity ${{match.traits.payload_kg}}kg &ge; ${{val}}kg)` : `Exact Match (${{match.traits.payload_kg}}kg == ${{val}}kg)`;
        resBox.innerHTML = `<strong>✅ REALIZABLE!</strong> Payload of ${{val}} kg is physically satisfied by coalition <strong>${{match.name}}</strong> [${{modeText}}].`;
      }} else {{
        resBox.style.background = "rgba(244, 63, 94, 0.15)";
        resBox.style.border = "1px solid rgba(244, 63, 94, 0.4)";
        resBox.style.color = "#FB7185";
        const reason = allowSurplus ? `Exceeds max fleet capacity (${{Math.max(...realizablePayloads)}} kg)` : `No coalition has an exact ${{val}} kg trait vector (available: [${{realizablePayloads.join(", ")}}] kg)`;
        resBox.innerHTML = `<strong>❌ UNREALIZABLE!</strong> ${{reason}}. Continuous MRTA optimizers would fail here!`;
      }}
    }}

    // Init
    renderMap();
    renderTabContent();
    selectTask("T4");
  </script>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML dashboard to: {html_path}")


def main():
    TerminalDisplay.print_header(
        "E-ITAGS: TraitEngine & WarehouseVisualizer Live Demonstration",
        "Efficacy-Optimized Multi-Robot Task Allocation in Autonomous Warehouses"
    )

    config_path = Path(__file__).parent / "configs" / "warehouse_scenario.yaml"
    scenario_name, robots, tasks = load_warehouse_scenario(config_path)

    print(f"\n📂 Scenario: '{scenario_name}'")
    print(f"   Fleet: {len(robots)} Heterogeneous Robots | Tasks: {len(tasks)} Inbound/Sorting/Docking Tasks\n")

    # Step 1: TraitEngine Coalition Enumeration
    TerminalDisplay.print_section("1. COALITION ENUMERATION & PHYSICAL TRAIT AGGREGATION")
    coalitions = TraitEngine.enumerate_all_coalitions(robots, max_size=3)
    print(f"From {len(robots)} robots, TraitEngine enumerated {len(coalitions)} non-empty coalitions up to size 3:\n")
    TerminalDisplay.display_coalitions(coalitions)

    # Step 2: Realizability Analysis
    TerminalDisplay.print_section("2. REALIZABILITY ANALYSIS (Why Continuous Optimizers Fail)")
    print("In classical optimization, algorithms hypothesize continuous traits (e.g., 500kg or 1500kg).")
    print("E-ITAGS computes the exact discrete realizable trait set Ω from physical robots.\n")

    query_payloads = [0.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0]
    for p in query_payloads:
        target = RobotTraits(payload_kg=p)
        # 1. Exact realizability check (allow_surplus=False)
        is_exact, match_exact = TraitEngine.is_realizable(target, robots, trait_subset=["payload_kg"], allow_surplus=False)
        # 2. Threshold coverage check (allow_surplus=True)
        is_surplus, match_surplus = TraitEngine.is_realizable(target, robots, trait_subset=["payload_kg"], allow_surplus=True)

        exact_str = f"Exact: {'✅ REAL' if is_exact else '❌ UNREAL'}"
        surplus_str = f"Surplus (≥): {'✅ SATISFIED by ' + str(match_surplus) if is_surplus else '❌ EXCEEDS FLEET'}"
        print(f"  • Payload {p:>5.0f} kg  ->  {exact_str:<18} | {surplus_str}")

    # Step 3: Trait Satisfaction Matrix
    TerminalDisplay.print_section("3. TRAIT SATISFACTION MATRIX S(C_k, t_j)")
    TerminalDisplay.display_satisfaction_matrix(coalitions, tasks)

    # Step 4: Optimal Coalition Allocation
    TerminalDisplay.print_section("4. OPTIMAL COALITION ASSIGNMENTS")
    allocations: dict[str, Coalition] = {}
    satisfaction_scores: dict[str, float] = {}

    for task in tasks:
        # Find capable coalitions with score = 1.0 (or highest score)
        best_coalition = None
        best_score = -1.0
        min_size = 999

        for c in coalitions:
            score, _ = TraitEngine.compute_trait_satisfaction(c, task)
            # Prefer higher satisfaction, then smaller coalition size (efficiency)
            if score > best_score or (score == best_score and c.size < min_size):
                best_score = score
                best_coalition = c
                min_size = c.size

        if best_coalition:
            allocations[task.id] = best_coalition
            satisfaction_scores[task.id] = best_score
            status = "FULL SATISFACTION" if best_score == 1.0 else f"PARTIAL ({best_score:.2f})"
            print(f"  • Task {task.id:<3} ('{task.name:<28}') -> Assigned: {str(best_coalition):<30} [{status}]")

    # Step 5: WarehouseVisualizer Export
    TerminalDisplay.print_section("5. EXPORTING VISUALIZATION ARTIFACTS (WarehouseVisualizer)")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    warehouse = WarehouseGrid()
    viz = WarehouseVisualizer(warehouse)

    # 5.1 Standard Map
    map_png = output_dir / "warehouse_map.png"
    viz.render(robots=robots, tasks=tasks, save_path=map_png)

    # 5.2 Allocation & Dispatch Map
    alloc_png = output_dir / "warehouse_allocation_map.png"
    viz.render_allocation_map(
        robots=robots,
        tasks=tasks,
        allocations=allocations,
        satisfaction_scores=satisfaction_scores,
        save_path=alloc_png
    )

    # 5.3 Satisfaction Heatmap
    matrix_png = output_dir / "trait_satisfaction_matrix.png"
    viz.render_satisfaction_matrix(coalitions=coalitions, tasks=tasks, save_path=matrix_png)

    # 5.4 Realizability Comparison Plot
    real_png = output_dir / "realizability_space.png"
    viz.render_realizability_chart(robots=robots, query_payloads=query_payloads, save_path=real_png)

    # 5.5 Collaborative Synergy Breakdown for Task T4
    collab_png = output_dir / "trait_synergy_t4.png"
    t4 = next(t for t in tasks if t.id == "T4")
    # Pick Single Forklift, Single Drone, and Combined Coalition
    fl_coalition = Coalition([robots[0]])
    dr_coalition = Coalition([robots[1]])
    fl_dr_coalition = Coalition([robots[0], robots[1]])
    viz.render_trait_breakdown(task=t4, coalitions=[fl_coalition, dr_coalition, fl_dr_coalition], save_path=collab_png)

    # 5.6 Comprehensive 4-Panel Research Dashboard
    dashboard_png = output_dir / "warehouse_trait_dashboard.png"
    viz.render_comprehensive_dashboard(robots=robots, tasks=tasks, allocations=allocations, save_path=dashboard_png)

    # 5.7 Interactive HTML Dashboard
    html_file = output_dir / "interactive_demo.html"
    export_interactive_html(warehouse, robots, tasks, coalitions, allocations, html_file)

    TerminalDisplay.print_header(
        "DEMONSTRATION COMPLETE",
        f"Generated 6 high-resolution visualization figures and interactive web dashboard in:\n  {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
