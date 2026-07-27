import os
import glob
import pandas as pd
from sklearn.ensemble import IsolationForest

from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

DB_DIR = "./chroma_db"
ARCHIVE_DIR = "./archive"

# Read API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# --- TOOL 1: Sensor Data Analyzer ---
@tool(description="Analyzes time-series sensor telemetry for a specific Turbofan Engine Unit across dataset files. Valid Unit IDs are integers between 1 and 260.")
def analyze_engine_sensors(unit_id: int) -> str:
    """
    Analyzes time-series sensor data for a specific Turbofan Engine Unit across all dataset files.
    Returns detected anomaly flags and current sensor averages.
    """
    file_paths = glob.glob(os.path.join(ARCHIVE_DIR, "train_FD*.txt"))
    if not file_paths:
        return "Error: No dataset files found in archive directory."

    col_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    col_names += [f'sensor_{i}' for i in range(1, 22)]

    df_list = []
    for file in file_paths:
        temp_df = pd.read_csv(file, sep=r'\s+', header=None, names=col_names)
        temp_df['source'] = os.path.basename(file)
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)

    # Filter for requested unit
    unit_data = combined_df[combined_df['unit_number'] == unit_id]
    if unit_data.empty:
        return f"No records found for Engine Unit #{unit_id}. Valid range across datasets is 1 to 260."

    sensor_cols = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11']
    model = IsolationForest(contamination=0.03, random_state=42)
    unit_data['is_anomaly'] = model.fit_predict(unit_data[sensor_cols])

    anomalies = unit_data[unit_data['is_anomaly'] == -1]
    max_cycle = unit_data['time_in_cycles'].max()

    avg_s2 = unit_data['sensor_2'].mean()
    avg_s3 = unit_data['sensor_3'].mean()
    avg_s4 = unit_data['sensor_4'].mean()
    avg_s11 = unit_data['sensor_11'].mean()

    return (
        f"Engine Unit #{unit_id} Analysis Summary:\n"
        f"- Max Cycle Reached: {max_cycle}\n"
        f"- Anomaly Events Flagged: {len(anomalies)}\n"
        f"- Telemetry Snapshot: Sensor 2={avg_s2:.2f}, Sensor 3={avg_s3:.2f}, "
        f"Sensor 4={avg_s4:.2f}, Sensor 11={avg_s11:.2f}"
    )


# --- TOOL 2: RAG Manual Search ---
@tool(description="Searches official equipment manuals and returns troubleshooting steps, error codes, and maintenance procedures matching the query.")
def search_equipment_manuals(query: str) -> str:
    """
    Searches official equipment manuals and returns troubleshooting steps,
    error codes, and maintenance procedures matching the query.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    results = vector_db.similarity_search(query, k=2)

    context = []
    for doc in results:
        src = doc.metadata.get('source', 'Manual')
        pg = doc.metadata.get('page', 'N/A')
        context.append(f"[Source: {src}, Page: {pg}]\n{doc.page_content.strip()}")

    return "\n\n".join(context)


# --- RUNNING THE AGENT ---
def run_maintenance_agent(unit_id: int):
    print(f"\n🤖 Initializing AI Agent for Engine Unit #{unit_id}...")

    if not GROQ_API_KEY:
        print("⚠️ GROQ_API_KEY environment variable not set!")
        print("Set it in terminal: $env:GROQ_API_KEY=\"gsk_...\"")
        return

    # 1. Register tools
    tools = [analyze_engine_sensors, search_equipment_manuals]

    # 2. Instantiate LLM
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    # 3. Create LangGraph ReAct Agent
    agent_executor = create_react_agent(llm, tools)

    # 4. Dynamic User Prompt
    user_request = (
        f"Check the status of Turbofan Engine Unit #{unit_id}. "
        f"If sensor anomalies are detected, search the manuals for matching troubleshooting steps "
        f"and generate a final text Maintenance Action Ticket."
    )

    print(f"💬 User Request: \"{user_request}\"\n")
    print("🔄 Agent thinking and executing tools...\n")

    # 5. Execute Agent Stream
    events = agent_executor.stream(
        {"messages": [("user", user_request)]},
        stream_mode="values"
    )

    final_answer = ""
    for event in events:
        message = event["messages"][-1]
        if message.type == "ai":
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"🛠️ Agent Called Tool: [{tool_call['name']}] args={tool_call['args']}")
            else:
                final_answer = message.content
        elif message.type == "tool":
            print(f"📥 Tool Response Received ({len(str(message.content))} chars)")

    print("\n" + "=" * 20 + " FINAL MAINTENANCE TICKET " + "=" * 20 + "\n")
    print(final_answer)


if __name__ == "__main__":
    print("\n---------------------------------------------------------")
    print("📋 NASA CMAPSS Dataset Unit Ranges:")
    print("   • FD001 & FD003: Units 1 - 100")
    print("   • FD004: Units 1 - 248")
    print("   • FD002: Units 1 - 260")
    print("   (Recommended Test Units: 1, 10, 42, 118, 133, 173, 250)")
    print("---------------------------------------------------------\n")

    user_input = input("Enter Turbofan Engine Unit ID to diagnose (1-260): ").strip()

    if user_input.isdigit() and 1 <= int(user_input) <= 260:
        target_unit = int(user_input)
    else:
        print("💡 Invalid or empty input. Defaulting to Engine Unit #42...")
        target_unit = 42

    run_maintenance_agent(unit_id=target_unit)