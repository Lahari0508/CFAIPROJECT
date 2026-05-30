"""
╔══════════════════════════════════════════════════════════════╗
║     WAREHOUSE ROBOT TASK SCHEDULER — CSP + AI ENGINE        ║
║     Constraint Satisfaction Problem with Backtracking        ║
║     MRV Heuristic | LCV Heuristic | Bayesian Uncertainty     ║
╚══════════════════════════════════════════════════════════════╝
"""

import heapq
import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. DATA MODELS — Environment Representation & State Space
# ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    location: str
    item: str
    deadline: int          # time unit deadline
    priority: int          # 1 = highest
    duration: int          # time units required
    assigned_robot: Optional[str] = None
    start_time: Optional[int] = None

    def __repr__(self):
        return (f"Task({self.id}: {self.item}@{self.location}, "
                f"deadline={self.deadline}, priority={self.priority})")


@dataclass
class Robot:
    id: str
    location: str
    battery: int           # 0–100
    max_battery: int = 100
    status: str = "idle"   # idle | busy | charging
    current_task: Optional[str] = None
    schedule: list = field(default_factory=list)

    def is_available(self):
        return self.status == "idle" and self.battery > 20

    def battery_ok(self):
        return self.battery > 20

    def __repr__(self):
        return (f"Robot({self.id} at {self.location}, "
                f"battery={self.battery}%, status={self.status})")


@dataclass
class WarehouseState:
    """Full state space snapshot."""
    robots: dict            # robot_id -> Robot
    tasks: dict             # task_id -> Task
    traffic: dict           # location -> congestion_level (0.0–1.0)
    time: int = 0

    def clone(self):
        import copy
        return copy.deepcopy(self)


# ─────────────────────────────────────────────────────────────
# 2. CONSTRAINT ENGINE — CSP Core
# ─────────────────────────────────────────────────────────────

class ConstraintEngine:
    """
    Core CSP engine:
    - Hard constraints:  no collisions, battery limits, deadline
    - Soft constraints:  minimize travel, prefer high-priority tasks
    """

    def __init__(self, state: WarehouseState):
        self.state = state

    def hard_constraints(self, robot: Robot, task: Task, start_time: int) -> tuple[bool, str]:
        """Returns (satisfied, reason)."""
        # Battery check
        travel_cost = self._travel_cost(robot.location, task.location)
        if robot.battery < travel_cost + 10:
            return False, f"LOW_BATTERY: {robot.id} battery={robot.battery}%"

        # Deadline check
        finish_time = start_time + task.duration + travel_cost
        if finish_time > task.deadline:
            return False, f"DEADLINE_MISS: finishes at {finish_time} > deadline {task.deadline}"

        # Collision / availability check
        for other_robot in self.state.robots.values():
            if other_robot.id == robot.id:
                continue
            if (other_robot.location == task.location and
                    other_robot.status == "busy"):
                return False, f"COLLISION: {other_robot.id} already at {task.location}"

        return True, "OK"

    def soft_constraints(self, robot: Robot, task: Task) -> float:
        """Returns a utility score (higher = better assignment)."""
        travel = self._travel_cost(robot.location, task.location)
        priority_bonus = (5 - task.priority) * 20    # priority 1 → +80
        battery_penalty = max(0, 30 - robot.battery) # penalize low battery
        congestion = self.state.traffic.get(task.location, 0.0) * 30
        return 100 - travel * 5 + priority_bonus - battery_penalty - congestion

    def _travel_cost(self, src: str, dst: str) -> int:
        """Simple Manhattan-style grid cost."""
        locations = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3",
                     "CHARGING", "DISPATCH"]
        src_idx = locations.index(src) if src in locations else 5
        dst_idx = locations.index(dst) if dst in locations else 5
        row_diff = abs(src_idx // 3 - dst_idx // 3)
        col_diff = abs(src_idx % 3 - dst_idx % 3)
        return (row_diff + col_diff) * 2 + 1


# ─────────────────────────────────────────────────────────────
# 3. HEURISTICS — MRV & LCV
# ─────────────────────────────────────────────────────────────

class Heuristics:

    @staticmethod
    def mrv(unassigned_tasks: list[Task], state: WarehouseState,
            engine: ConstraintEngine) -> Task:
        """
        Minimum Remaining Values:
        Pick the task with fewest valid robot assignments (most constrained first).
        """
        def count_valid(task):
            count = 0
            for robot in state.robots.values():
                ok, _ = engine.hard_constraints(robot, task, state.time)
                if ok:
                    count += 1
            return count

        return min(unassigned_tasks, key=count_valid)

    @staticmethod
    def lcv(task: Task, candidates: list[Robot],
            state: WarehouseState, engine: ConstraintEngine) -> list[Robot]:
        """
        Least Constraining Value:
        Order robots so that choosing them rules out fewest future assignments.
        """
        def freedom_score(robot):
            # How many OTHER tasks can this robot still do after this assignment?
            score = 0
            for t in state.tasks.values():
                if t.id == task.id or t.assigned_robot:
                    continue
                ok, _ = engine.hard_constraints(robot, t, state.time + task.duration)
                if ok:
                    score += 1
            return -score  # ascending → least constraining first

        return sorted(candidates, key=freedom_score)

    @staticmethod
    def deadline_priority(task: Task) -> int:
        """Earliest Deadline First tie-breaker."""
        return task.deadline * 10 + task.priority


# ─────────────────────────────────────────────────────────────
# 4. BAYESIAN UNCERTAINTY MODULE — Bottleneck Diagnosis
# ─────────────────────────────────────────────────────────────

class BayesianModule:
    """
    Models uncertainty via Bayesian belief updates.
    Tracks congestion probability at each location.
    """

    def __init__(self, locations: list[str]):
        # Prior: uniform belief over congestion levels
        self.beliefs = {loc: 0.1 for loc in locations}

    def observe(self, location: str, congestion_observed: float):
        """Update belief using simplified Bayesian update."""
        prior = self.beliefs[location]
        likelihood = congestion_observed  # P(obs | congested)
        # Bayes: posterior ∝ likelihood × prior
        posterior = (likelihood * prior) / max(
            likelihood * prior + (1 - likelihood) * (1 - prior), 1e-9)
        self.beliefs[location] = round(posterior, 3)

    def get_congestion(self, location: str) -> float:
        return self.beliefs.get(location, 0.1)

    def markov_transition(self):
        """Simulate time-step transition: congestion decays naturally."""
        for loc in self.beliefs:
            self.beliefs[loc] = max(0.05, self.beliefs[loc] * 0.9)

    def bottleneck_locations(self, threshold=0.5) -> list[str]:
        return [loc for loc, belief in self.beliefs.items()
                if belief >= threshold]

    def report(self):
        print("\n📡 Bayesian Congestion Beliefs:")
        for loc, belief in sorted(self.beliefs.items(),
                                  key=lambda x: -x[1]):
            bar = "█" * int(belief * 20)
            print(f"  {loc:10s} [{bar:<20s}] {belief:.1%}")


# ─────────────────────────────────────────────────────────────
# 5. BACKTRACKING SEARCH ENGINE (DFS + CSP)
# ─────────────────────────────────────────────────────────────

class BacktrackingScheduler:
    """
    DFS-based backtracking search over the assignment space.
    Variables = tasks, Domains = robots × start_times
    """

    def __init__(self, state: WarehouseState):
        self.state = state
        self.engine = ConstraintEngine(state)
        self.stats = {"nodes": 0, "backtracks": 0, "pruned": 0}

    def solve(self) -> Optional[dict]:
        unassigned = [t for t in self.state.tasks.values()
                      if not t.assigned_robot]
        assignment = {}
        result = self._backtrack(unassigned, assignment)
        self._print_stats()
        return result

    def _backtrack(self, unassigned: list[Task],
                   assignment: dict) -> Optional[dict]:
        if not unassigned:
            return assignment  # ✅ Complete assignment found

        self.stats["nodes"] += 1

        # MRV: choose most constrained task
        task = Heuristics.mrv(unassigned, self.state, self.engine)
        remaining = [t for t in unassigned if t.id != task.id]

        # Gather candidate robots
        candidates = [r for r in self.state.robots.values()
                      if r.is_available()]

        # LCV: order by least constraining
        ordered = Heuristics.lcv(task, candidates, self.state, self.engine)

        for robot in ordered:
            ok, reason = self.engine.hard_constraints(
                robot, task, self.state.time)

            if not ok:
                self.stats["pruned"] += 1
                continue  # Prune branch

            # Assign
            assignment[task.id] = robot.id
            robot.status = "busy"
            robot.current_task = task.id
            task.assigned_robot = robot.id
            task.start_time = self.state.time

            result = self._backtrack(remaining, assignment)
            if result is not None:
                return result  # ✅ Found solution

            # Backtrack
            self.stats["backtracks"] += 1
            del assignment[task.id]
            robot.status = "idle"
            robot.current_task = None
            task.assigned_robot = None
            task.start_time = None

        return None  # ❌ No valid assignment

    def _print_stats(self):
        print(f"\n🔍 Search Statistics:")
        print(f"  Nodes explored : {self.stats['nodes']}")
        print(f"  Backtracks     : {self.stats['backtracks']}")
        print(f"  Branches pruned: {self.stats['pruned']}")


# ─────────────────────────────────────────────────────────────
# 6. ADVERSARIAL / PREFERENCE DECISION MAKING
# ─────────────────────────────────────────────────────────────

class PreferenceDecisionMaker:
    """
    Utility-based soft constraint optimizer.
    Ranks final schedules by multi-criteria utility.
    """

    WEIGHTS = {
        "deadline_met":      50,
        "travel_efficiency": 20,
        "battery_preserved": 15,
        "priority_served":   15,
    }

    def evaluate(self, assignment: dict, state: WarehouseState,
                 engine: ConstraintEngine) -> float:
        score = 0.0
        for task_id, robot_id in assignment.items():
            task = state.tasks[task_id]
            robot = state.robots[robot_id]
            travel = engine._travel_cost(robot.location, task.location)
            finish = (task.start_time or 0) + task.duration + travel

            if finish <= task.deadline:
                score += self.WEIGHTS["deadline_met"]
            score += self.WEIGHTS["travel_efficiency"] * (1 / max(travel, 1))
            score += self.WEIGHTS["battery_preserved"] * (robot.battery / 100)
            score += self.WEIGHTS["priority_served"] * (1 / task.priority)

        return round(score, 2)


# ─────────────────────────────────────────────────────────────
# 7. INTEGRATED PIPELINE — Main Scheduler
# ─────────────────────────────────────────────────────────────

class WarehouseRobotScheduler:
    """
    Full integrated pipeline:
    Input Parameters → CSP Engine → Schedule Output + Reasoning Trace
    """

    LOCATIONS = ["A1", "A2", "A3", "B1", "B2", "B3",
                 "C1", "C2", "C3", "CHARGING", "DISPATCH"]

    def __init__(self):
        self.bayesian = BayesianModule(self.LOCATIONS)
        self.state = None
        self.trace = []

    def setup(self, robots: list[Robot], tasks: list[Task],
              traffic: dict = None):
        self.state = WarehouseState(
            robots={r.id: r for r in robots},
            tasks={t.id: t for t in tasks},
            traffic=traffic or {}
        )
        # Sync Bayesian beliefs with observed traffic
        for loc, cong in (traffic or {}).items():
            self.bayesian.observe(loc, cong)

    def run(self) -> dict:
        print("=" * 62)
        print("  🤖 WAREHOUSE ROBOT TASK SCHEDULER — CSP PIPELINE")
        print("=" * 62)

        self._print_state()

        # Bayesian bottleneck analysis
        bottlenecks = self.bayesian.bottleneck_locations(0.4)
        if bottlenecks:
            print(f"\n⚠️  Bottleneck locations detected: {bottlenecks}")
            self.trace.append(f"Bottlenecks: {bottlenecks}")

        # Run backtracking CSP solver
        print("\n🔄 Running Backtracking Search (DFS + CSP)...")
        scheduler = BacktrackingScheduler(self.state)
        assignment = scheduler.solve()

        if not assignment:
            print("\n❌ No complete schedule found! Partial fallback applied.")
            assignment = self._greedy_fallback()

        # Evaluate with preference decision maker
        pdm = PreferenceDecisionMaker()
        engine = ConstraintEngine(self.state)
        utility = pdm.evaluate(assignment, self.state, engine)

        # Build output schedule
        schedule = self._build_schedule(assignment)

        print("\n📋 FINAL SCHEDULE:")
        print("-" * 62)
        for entry in schedule:
            print(f"  ✅ {entry['task_id']:6s} → Robot {entry['robot']:4s} | "
                  f"Start T={entry['start']:2d} | "
                  f"Location: {entry['location']:10s} | "
                  f"Item: {entry['item']}")

        print(f"\n💡 Utility Score: {utility}")
        self.bayesian.report()
        self._reasoning_trace(assignment)

        return {"assignment": assignment, "schedule": schedule,
                "utility": utility, "trace": self.trace}

    def _greedy_fallback(self) -> dict:
        """Simple greedy fallback for unsolvable CSP."""
        assignment = {}
        engine = ConstraintEngine(self.state)
        robots = list(self.state.robots.values())
        tasks = sorted(self.state.tasks.values(),
                       key=lambda t: (t.deadline, t.priority))
        robot_idx = 0
        for task in tasks:
            if robot_idx < len(robots):
                r = robots[robot_idx]
                ok, _ = engine.hard_constraints(r, task, self.state.time)
                if ok:
                    assignment[task.id] = r.id
                    task.assigned_robot = r.id
                    task.start_time = self.state.time
                    r.status = "busy"
                    robot_idx += 1
        return assignment

    def _build_schedule(self, assignment: dict) -> list[dict]:
        schedule = []
        engine = ConstraintEngine(self.state)
        for task_id, robot_id in assignment.items():
            task = self.state.tasks[task_id]
            robot = self.state.robots[robot_id]
            travel = engine._travel_cost(robot.location, task.location)
            schedule.append({
                "task_id": task_id,
                "robot": robot_id,
                "item": task.item,
                "location": task.location,
                "start": task.start_time or 0,
                "finish": (task.start_time or 0) + task.duration + travel,
                "deadline": task.deadline,
            })
        return sorted(schedule, key=lambda x: x["start"])

    def _reasoning_trace(self, assignment: dict):
        print("\n🧠 REASONING TRACE:")
        print("-" * 62)
        engine = ConstraintEngine(self.state)
        for task_id, robot_id in assignment.items():
            task = self.state.tasks[task_id]
            robot = self.state.robots[robot_id]
            score = engine.soft_constraints(robot, task)
            travel = engine._travel_cost(robot.location, task.location)
            reason = (f"  {task_id} → {robot_id}: "
                      f"utility={score:.1f}, travel={travel}, "
                      f"battery={robot.battery}%")
            print(reason)
            self.trace.append(reason)

    def _print_state(self):
        print(f"\n📦 Tasks ({len(self.state.tasks)}):")
        for t in sorted(self.state.tasks.values(), key=lambda x: x.priority):
            print(f"  {t}")
        print(f"\n🤖 Robots ({len(self.state.robots)}):")
        for r in self.state.robots.values():
            print(f"  {r}")


# ─────────────────────────────────────────────────────────────
# 8. SCALABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────

def scalability_analysis():
    """Measure how CSP scales with increasing tasks/robots."""
    import time
    print("\n" + "=" * 62)
    print("  📈 SCALABILITY ANALYSIS")
    print("=" * 62)
    locations = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"]
    items = ["Box", "Package", "Pallet", "Crate", "Bin"]

    results = []
    for n_robots in [2, 4, 6]:
        for n_tasks in [3, 6, 9, 12]:
            robots = [
                Robot(f"SC{i+1}", random.choice(locations),
                      random.randint(40, 100))
                for i in range(n_robots)
            ]
            tasks = [
                Task(f"T{i+1}", random.choice(locations),
                     random.choice(items),
                     deadline=random.randint(10, 30),
                     priority=random.randint(1, 4),
                     duration=random.randint(1, 5))
                for i in range(n_tasks)
            ]
            sched = WarehouseRobotScheduler()
            sched.setup(robots, tasks, {})
            t0 = time.perf_counter()
            scheduler = BacktrackingScheduler(sched.state)
            result = scheduler.solve()
            elapsed = (time.perf_counter() - t0) * 1000
            assigned = len(result) if result else 0
            results.append((n_robots, n_tasks, elapsed, assigned))
            print(f"  Robots={n_robots:2d} Tasks={n_tasks:2d} → "
                  f"Assigned={assigned:2d}/{n_tasks:2d}  "
                  f"Time={elapsed:6.1f}ms")

    print("\n⚡ Observation: CSP with MRV+LCV scales near-linearly for")
    print("   small-medium instances. Large instances benefit from")
    print("   local search (min-conflicts) or constraint propagation.")
    return results


# ─────────────────────────────────────────────────────────────
# 9. DEMO SCENARIO
# ─────────────────────────────────────────────────────────────

def main():
    robots = [
        Robot("SC1", "A1", battery=85),
        Robot("SC2", "B2", battery=45),
        Robot("SC3", "C1", battery=92),
        Robot("SC4", "A3", battery=30),
    ]

    tasks = [
        Task("T001", "B1", "Electronics Box",   deadline=15, priority=1, duration=3),
        Task("T002", "A2", "Heavy Pallet",       deadline=20, priority=2, duration=5),
        Task("T003", "C3", "Fragile Package",    deadline=12, priority=1, duration=2),
        Task("T004", "B3", "Chemical Bin",       deadline=25, priority=3, duration=4),
        Task("T005", "A1", "Priority Crate",     deadline=10, priority=1, duration=2),
        Task("T006", "C2", "Standard Box",       deadline=30, priority=4, duration=3),
    ]

    traffic = {
        "B1": 0.7,   # High congestion
        "C3": 0.3,
        "A2": 0.5,
    }

    scheduler = WarehouseRobotScheduler()
    scheduler.setup(robots, tasks, traffic)
    result = scheduler.run()

    # Scalability study
    scalability_analysis()

    print("\n" + "=" * 62)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 62)


if __name__ == "__main__":
    main()
