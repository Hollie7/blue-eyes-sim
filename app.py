# streamlit_app.py
import streamlit as st
import pandas as pd
from simulator import BlueEyesWorld

"""
A simple Streamlit UI for the blue-eyed island LLM agent simulation.

Features:
- Choose number of blue-eyed and red-eyed agents.
- Choose how many days to simulate.
- View final summary (who left on which day).
- Inspect each agent's reasoning process day by day.
"""


def run_simulation(n_blue, n_red, max_days, model):
    """
    Helper function to run one simulation and return the result dictionary.
    """
    world = BlueEyesWorld(n_blue=n_blue, n_red=n_red, model=model)
    result = world.run(max_days=max_days, verbose=False)
    return result


def main():
    st.set_page_config(
        page_title="Blue-Eyed Island LLM Simulation",
        layout="wide",
    )

    st.title("🧠 Blue-Eyed Island – LLM Multi-Agent Simulation")
    st.markdown(
        """
This app simulates the classic *blue-eyed islanders* puzzle using LLM-based agents.

Each agent:
- Sees other agents' eye colors but not their own.
- Knows the public statement: *"There is at least one blue-eyed person."*
- Is assumed to be perfectly rational.
- Must leave the island the night they logically deduce their own eye color.

You can:
- Choose the number of blue- and red-eyed agents.
- Choose how many days to simulate.
- Inspect each agent's reasoning on each day.
        """
    )

    st.sidebar.header("Simulation Settings")

    # Simulation parameters
    n_blue = st.sidebar.number_input(
        "Number of blue-eyed agents",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    n_red = st.sidebar.number_input(
        "Number of red-eyed agents",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
    )

    max_days = st.sidebar.number_input(
        "Max simulation days",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )

    model = st.sidebar.selectbox(
        "OpenAI model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Use a cheaper model for quick experiments, or a stronger one for more robust reasoning.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Tip:** Start with small numbers (e.g., 2–3 blue agents) to keep the simulation fast."
    )

    # Run button
    if st.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation with LLM agents..."):
            sim_result = run_simulation(
                n_blue=int(n_blue),
                n_red=int(n_red),
                max_days=int(max_days),
                model=model,
            )

        # Store in session_state so we can interact with it across reruns
        st.session_state["sim_result"] = sim_result
        st.success("Simulation completed!")

    # If we already have a result, show it
    if "sim_result" in st.session_state:
        sim_result = st.session_state["sim_result"]

        st.header("📊 Simulation Summary")

        # Summary table: one row per agent
        agents_df = pd.DataFrame(sim_result["agents"])
        agents_df = agents_df.sort_values("id")

        st.subheader("Agents overview")
        st.dataframe(
            agents_df.rename(columns={
                "id": "Agent ID",
                "color": "Eye color",
                "left_day": "Departure day",
            }),
            use_container_width=True,
        )

        st.markdown(
            """
- **Departure day** is the day an agent decided to leave the island.
- `NaN` means the agent never left within the simulated days.
            """
        )

        # Day selector and reasoning view
        st.header("🧩 Reasoning by Day and Agent")

        day_logs = sim_result["day_logs"]
        available_days = sorted(day_logs.keys())
        selected_day = st.selectbox(
            "Select a day to inspect",
            options=available_days,
            format_func=lambda d: f"Day {d}",
        )

        day_log = day_logs.get(selected_day, {})

        st.subheader(f"Day {selected_day} – Agent decisions and reasoning")

        if not day_log:
            st.info("No logs for this day (no agent made a decision).")
        else:
            # Show per-agent reasoning using expanders
            for agent_id in sorted(day_log.keys()):
                info = day_log[agent_id]
                color = info["color"]
                decision = info["decision"]
                reasoning = info["reasoning"]

                title = f"Agent {agent_id} ({color} eyes) – decision: {decision}"
                with st.expander(title, expanded=(decision == "LEAVE")):
                    st.markdown(
                        f"**Agent ID:** {agent_id}  \n"
                        f"**Eye color (ground truth):** `{color}`  \n"
                        f"**Decision:** `{decision}`"
                    )
                    st.markdown("**Reasoning:**")
                    st.write(reasoning)

        # Optional: show raw structure for debugging
        with st.expander("🔍 Raw day logs (debug view)"):
            st.json(day_logs)


if __name__ == "__main__":
    main()
