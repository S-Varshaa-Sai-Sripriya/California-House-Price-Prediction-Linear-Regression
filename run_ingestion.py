from data_ingestion.load_data import ingest_data

if __name__ == "__main__":
    df = ingest_data()
    print(df.head())