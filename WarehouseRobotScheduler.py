"""
Warehouse Robot Task Scheduler
Uses CSP + Backtracking + Heuristics
Time Complexity: O(n * b^d) where n=tasks, b=branching factor, d=depth
"""

import random
from dataclasses import dataclass, field
from typing import Optional

# ─── Data Models ────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    location: str
    priority: int          # 1=high, 3=low
    deadline: int          # steps remaining
    weight: float          # load units

@dataclass
class Robot:
    id: str
    battery: int           # 0–100
    current_location: str
    max_load: float = 10.0
    current_load: float = 0.0

# ─── CSP Engine ─────────────────────────────────────────────────────────────

class WarehouseScheduler:
    LOCATIONS = ["A1","A2","B1","B2","C1","C2","DOCK"]

    def __init__(self, robots: list[Robot], tasks: list[Task]):
        self.robots = robots
        self.tasks  = tasks
        self.assignment: dict[str, str] = {}   # task_id → robot_id
        self.steps = 0

    # --- Constraints ---
    def _check(self, robot: Robot, task: Task, tentative_load: float) -> tuple[bool, str]:
        if robot.battery < 20:
            return False, "LOW_BATTERY"
        if tentative_load > robot.max_load:
            return False, "OVERLOAD"
        return True, "OK"

    # --- MRV heuristic: task with fewest valid robots first ---
    def _mrv_order(self) -> list[Task]:
        unassigned = [t for t in self.tasks if t.id not in self.assignment]
        def valid_count(task):
            return sum(1 for r in self.robots
                       if self._check(r, task, r.current_load + task.weight)[0])
        return sorted(unassigned, key=lambda t: (valid_count(t), t.deadline))

    # --- LCV heuristic: robot with closest deadline awareness ---
    def _lcv_order(self, task: Task) -> list[Robot]:
        def score(robot):
            dist = abs(self.LOCATIONS.index(robot.current_location) -
                       self.LOCATIONS.index(task.location))
            return dist + (3 - task.priority) * 2
        return sorted(self.robots, key=score)

    # --- Backtracking search (DFS) ---
    def _backtrack(self) -> bool:
        self.steps += 1
        unassigned = [t for t in self.tasks if t.id not in self.assignment]
        if not unassigned:
            return True                          # all tasks assigned ✓

        task = self._mrv_order()[0]
        for robot in self._lcv_order(task):
            new_load = robot.current_load + task.weight
            ok, reason = self._check(robot, task, new_load)
            if ok:
                # assign
                self.assignment[task.id] = robot.id
                old_load = robot.current_load
                robot.current_load = new_load

                if self._backtrack():
                    return True

                # undo
                del self.assignment[task.id]
                robot.current_load = old_load

        return False                             # backtrack

    def solve(self) -> dict:
        success = self._backtrack()
        schedule = []
        for task in self.tasks:
            rid = self.assignment.get(task.id, "UNASSIGNED")
            robot = next((r for r in self.robots if r.id == rid), None)
            schedule.append({
                "task":     task.id,
                "location": task.location,
                "priority": task.priority,
                "robot":    rid,
                "battery":  robot.battery if robot else "—",
            })
        return {"success": success, "schedule": schedule, "steps": self.steps}


# ─── Demo Run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    locs = WarehouseScheduler.LOCATIONS

    robots = [
        Robot("R1", battery=85, current_location="A1"),
        Robot("R2", battery=60, current_location="B2"),
        Robot("R3", battery=30, current_location="DOCK"),
    ]

    tasks = [
        Task("T1", "A2", priority=1, deadline=3, weight=4.0),
        Task("T2", "C1", priority=2,
             deadline=5, weight=6.0),
        Task("T3", "B1", priority=1, deadline=2, weight=3.5),
        Task("T4", "DOCK",priority=3,deadline=8, weight=2.0),
        Task("T5", "A1", priority=2, deadline=4, weight=5.0),
    ]

    scheduler = WarehouseScheduler(robots, tasks)
    result = scheduler.solve()

    print("=" * 50)
    print("  WAREHOUSE ROBOT TASK SCHEDULER — RESULTS")
    print("=" * 50)
    print(f"  Solved : {'✓ YES' if result['success'] else '✗ NO'}")
    print(f"  Steps  : {result['steps']}")
    print("-" * 50)
    print(f"  {'Task':<6} {'Location':<8} {'Priority':<10} {'Robot':<12} {'Battery'}")
    print("-" * 50)
    for row in result["schedule"]:
        p_label = ["", "HIGH", "MED", "LOW"][row["priority"]]
        print(f"  {row['task']:<6} {row['location']:<8} {p_label:<10} {row['robot']:<12} {row['battery']}%")
    print("=" * 50)
        