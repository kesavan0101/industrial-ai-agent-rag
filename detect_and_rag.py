import os
import glob
import pandas as pd
from sklearn.ensemble import IsolationForest
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "./chroma_db"
ARCHIVE_DIR = "./archive"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")


def load_all_dataset_files():
    file_paths = glob.glob(os.path.join(ARCHIVE_DIR, "train_FD*.txt"))
    if not file_paths:
        raise FileNotFoundError(f"No 'train_FD*.txt' files found in '{ARCHIVE_DIR}'.")

    col_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    col_names += [f'sensor_{i}' for i in range(1, 22)]

    df_list = []
    for file in file_paths:
        file_name = os.path.basename(file)
        print(f"  └─ Ingesting {file_name}...")
        temp_df = pd.read_csv(file, sep=r'\s+', header=None, names=col_names)
        temp_df['dataset_source'] = file_name
        df_list.append(temp_df)

    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combined {len(file_paths)} NASA files into {len(combined_df):,} total sensor records!\n")
    return combined_df


def analyze_and_rag_all():
    # 1. Load All Datasets
    print("📊 Loading multi-file sensor telemetry...")
    df = load_all_dataset_files()
    sensor_cols = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_11']

    # 2. Anomaly Detection across whole dataset
    print("🤖 Running Isolation Forest model on all records...")
    model = IsolationForest(contamination=0.02, random_state=42)
    df['is_anomaly'] = model.fit_predict(df[sensor_cols])

    anomalies = df[df['is_anomaly'] == -1]
    print(f"🚨 Total Anomalies Detected: {len(anomalies):,} out of {len(df):,} records.")

    # 3. Group and Rank Critical Incidents by Engine Unit & File
    # We find the maximum operating cycle reached per anomalous unit to flag critical failure stages
    critical_cases = (
        anomalies.groupby(['dataset_source', 'unit_number'])
        .agg(
            max_cycle=('time_in_cycles', 'max'),
            avg_sensor_2=('sensor_2', 'mean'),
            avg_sensor_3=('sensor_3', 'mean'),
            avg_sensor_4=('sensor_4', 'mean'),
            avg_sensor_11=('sensor_11', 'mean'),
            anomaly_count=('is_anomaly', 'count')
        )
        .reset_index()
        .sort_values(by='max_cycle', ascending=False)
    )

    print(f"📊 Identified {len(critical_cases)} distinct engines exhibiting abnormal wear patterns.")

    # Take Top 3 Most Severe Failure Cases to process
    top_critical = critical_cases.head(3)

    # 4. Initialize Vector DB
    print("\n🧠 Loading ChromaDB vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("\n" + "=" * 25 + " BATCH DIAGNOSTIC ANALYSIS REPORT " + "=" * 25 + "\n")

    # 5. Process RAG Queries for Each Critical Case
    for index, row in top_critical.iterrows():
        source_file = row['dataset_source']
        unit = int(row['unit_number'])
        cycle = int(row['max_cycle'])
        anom_count = int(row['anomaly_count'])

        query = (
            f"Turbofan Engine [{source_file}] Unit #{unit} near failure at cycle {cycle}. "
            f"Sensor metrics: sensor_2: {row['avg_sensor_2']:.2f}, sensor_3: {row['avg_sensor_3']:.2f}, "
            f"sensor_4: {row['avg_sensor_4']:.2f}, sensor_11: {row['avg_sensor_11']:.2f}. "
            f"Troubleshooting procedures, motor fault handling, and repair SOPs."
        )

        print(f"🔴 CRITICAL TARGET #{index + 1}: Engine Unit #{unit} ({source_file})")
        print(f"   ├─ Max Cycle Reached: {cycle}")
        print(f"   ├─ Total Anomaly Signals: {anom_count}")
        print(f"   └─ Search Query: \"{query}\"")

        # Retrieve matching manuals
        results = vector_db.similarity_search(query, k=2)

        print("\n   🔧 RETRIEVED MAINTENANCE MANUAL INSTRUCTIONS:")
        for i, doc in enumerate(results, 1):
            src = doc.metadata.get('source', 'Manual')
            pg = doc.metadata.get('page', 'N/A')
            snippet = doc.page_content.strip().replace("\n", " ")[:200]
            print(f"      [{i}] Ref: {src} (Pg {pg}) -> \"{snippet}...\"")
        print("-" * 75 + "\n")


if __name__ == "__main__":
    analyze_and_rag_all()