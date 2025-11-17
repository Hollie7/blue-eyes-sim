# blue_eyes_sim.py
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

# Load environment variables (OPENAI_API_KEY, etc.)
load_dotenv()

# If OPENAI_API_KEY is set in the environment, the default client will pick it up
client = OpenAI()


class Agent:
    """
    A simple LLM-based agent representing an islander.

    - Each agent knows:
        * its own ID (but not its own eye color)
        * other agents' eye colors (visible_info)
        * complete history of departures
    - Each day it calls the LLM once to decide whether to STAY or LEAVE.
    - The agent stores its reasoning for each day.
    """

    def __init__(self, agent_id, color, model="gpt-4o-mini"):
        self.id = agent_id
        self.color = color          # "blue" or "red" (ground truth, for evaluation only)
        self.model = model
        self.left_day = None        # day when the agent left the island (None = still on island)
        self.reasoning_by_day = {}  # {day: {"decision": str, "reasoning": str}}

    def see_world(self, all_agents):
        """
        Return the visible world from this agent's perspective:
        a list of other agents' IDs and eye colors.
        """
        visible = []
        for a in all_agents:
            if a.id == self.id:
                continue
            visible.append({"id": a.id, "color": a.color})
        return visible

    def make_decision(self, day, visible_info, history):
        """
        Call the LLM to decide whether to STAY or LEAVE on this day,
        based on:
        - current day index
        - visible eye colors of other agents
        - departure history so far.

        Returns:
            decision (str): "STAY" or "LEAVE"
            reasoning (str): natural language reasoning from the model
        """

        # Format departure history as text
        if not history:
            history_text = "No one has left the island in the previous days."
        else:
            lines = []
            for d, leavers in history.items():
                if leavers:
                    lines.append(
                        f"Day {d}: agents {', '.join(str(i) for i in leavers)} left the island."
                    )
                else:
                    lines.append(f"Day {d}: no one left."
                                 )
            history_text = "\n".join(lines)

        # Describe what this agent can see
        if visible_info:
            others_desc = ", ".join(
                [f"Agent {item['id']} has {item['color']} eyes" for item in visible_info]
            )
        else:
            others_desc = "You see no other agents."

        # System prompt to enforce strict format
        system_prompt = (
            "You are a perfectly rational logician in the classic blue-eyed island puzzle.\n"
            "Everyone knows the following rules:\n"
            "1. Each agent can see others' eye colors but not their own.\n"
            "2. It is common knowledge that there is at least one blue-eyed person on the island.\n"
            "3. All agents are perfectly rational and know that all others are perfectly rational (common knowledge of rationality).\n"
            "4. Every midnight, if an agent has logically deduced their own eye color with certainty, "
            "they must leave the island that night.\n"
            "5. Agents can observe who has left on previous days, but they never talk about eye colors.\n\n"
            "You MUST use classical logical reasoning and you MUST respond in strict JSON format:\n"
            "{\n"
            "  \"decision\": \"STAY\" or \"LEAVE\",\n"
            "  \"reasoning\": \"a short explanation of your logical reasoning\"\n"
            "}\n"
            "Do not add any extra keys or text outside the JSON. No markdown."
        )


        user_prompt = (
            f"Today is day {day}.\n"
            f"You are Agent {self.id}.\n"
            f"You can see the following other agents and their eye colors:\n"
            f"{others_desc}\n\n"
            f"Here is the full history of departures so far:\n"
            f"{history_text}\n\n"
            "Based on this information and the common knowledge assumptions, decide whether you have logically deduced "
            "your own eye color with certainty and must leave tonight, or you must stay.\n"
            "Again, reply ONLY with a JSON object with keys 'decision' and 'reasoning'."
        )

        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
        )

        raw_text = response.choices[0].message.content.strip()

        # Try to parse JSON
        decision = "STAY"
        reasoning = f"Failed to parse JSON. Raw response: {raw_text}"

        try:
            data = json.loads(raw_text)
            decision_raw = str(data.get("decision", "")).upper()
            if "LEAVE" in decision_raw:
                decision = "LEAVE"
            else:
                decision = "STAY"
            reasoning = str(data.get("reasoning", ""))
        except Exception:
            # Fallback: simple heuristic based on the text
            upper = raw_text.upper()
            if "LEAVE" in upper:
                decision = "LEAVE"
                reasoning = raw_text
            else:
                decision = "STAY"
                reasoning = raw_text

        # Store reasoning internally
        self.reasoning_by_day[day] = {
            "decision": decision,
            "reasoning": reasoning,
        }

        return decision, reasoning


class BlueEyesWorld:
    """
    A world that manages a group of LLM agents in the blue-eyed island puzzle.

    - It tracks:
        * agents and their true colors
        * day-by-day departure history
        * per-day reasoning logs for each agent
    """

    def __init__(self, n_blue, n_red, model="gpt-4o-mini"):
        self.agents = []
        agent_id = 1

        # Create blue-eyed agents
        for _ in range(n_blue):
            self.agents.append(Agent(agent_id, "blue", model=model))
            agent_id += 1

        # Create red-eyed agents
        for _ in range(n_red):
            self.agents.append(Agent(agent_id, "red", model=model))
            agent_id += 1

        self.day = 0
        # history: {day: [agent_ids_that_left]}
        self.history = {}
        # day_logs: {day: {agent_id: {decision, reasoning, color}}}
        self.day_logs = {}

    def step(self):
        """
        Simulate one day: each agent that has not yet left makes a decision.
        """
        self.day += 1
        today_leavers = []
        self.day_logs[self.day] = {}

        for agent in self.agents:
            if agent.left_day is not None:
                continue  # Skip agents who already left

            visible_info = agent.see_world(self.agents)
            decision, reasoning = agent.make_decision(self.day, visible_info, self.history)

            # Store detailed log for Streamlit frontend
            self.day_logs[self.day][agent.id] = {
                "decision": decision,
                "reasoning": reasoning,
                "color": agent.color,
            }

            if decision == "LEAVE":
                agent.left_day = self.day
                today_leavers.append(agent.id)

        self.history[self.day] = today_leavers
        return today_leavers

    def run(self, max_days=20, verbose=False):
        """
        Run the simulation for up to `max_days` days.

        We DO NOT stop just because no one left on a given day,
        because in the blue-eyed island puzzle the fact that
        nobody leaves is itself informative.

        We only stop early if:
        - all agents have already left the island.
        """
        if verbose:
            print(
                f"=== Start simulation: {self.count_color('blue')} blue, "
                f"{self.count_color('red')} red ==="
            )

        for _ in range(max_days):
            leavers = self.step()

            if verbose:
                if leavers:
                    print(f"Day {self.day}: agents {leavers} left the island.")
                else:
                    print(f"Day {self.day}: no one left.")

            # Stop early only if all agents have left
            if all(a.left_day is not None for a in self.agents):
                if verbose:
                    print("All agents have left the island. Stopping early.")
                break

        if verbose:
            print("\nFinal status:")
            for a in self.agents:
                print(f"Agent {a.id} ({a.color}) left on day {a.left_day}")

        result = {
            "day_logs": self.day_logs,
            "agents": [
                {"id": a.id, "color": a.color, "left_day": a.left_day}
                for a in self.agents
            ],
        }
        return result


    def count_color(self, color):
        """
        Count agents with a given eye color.
        """
        return sum(1 for a in self.agents if a.color == color)


if __name__ == "__main__":
    # Simple CLI test
    world = BlueEyesWorld(n_blue=2, n_red=2, model="gpt-4o-mini")
    sim_result = world.run(max_days=10, verbose=True)

    print("\n=== Day logs (short preview) ===")
    for day, logs in sim_result["day_logs"].items():
        print(f"\nDay {day}:")
        for aid, info in logs.items():
            print(f"  Agent {aid} ({info['color']}): {info['decision']}")
