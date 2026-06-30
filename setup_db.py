import sqlite3
import pandas as pd
import os
import glob
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import zipfile

DB_PATH = "olist.db"
CHROMA_PATH = "chroma_db"

# 🌟 THE COMPLETE ENTERPRISE METADATA MAP (ALL 9 TABLES) 🌟
OLIST_METADATA = {
    "customers": {
        "description": "Customer demographics including customer_id, unique id, zip code, city, and state. Joins with orders.",
        "ddl": """CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_unique_id TEXT,
            customer_zip_code_prefix INTEGER,
            customer_city TEXT,
            customer_state TEXT
        );"""
    },
    "geolocation": {
        "description": "Brazilian zip codes, latitude, longitude, city, and state. Used for mapping, distances, and location analysis.",
        "ddl": """CREATE TABLE geolocation (
            geolocation_zip_code_prefix INTEGER,
            geolocation_lat REAL,
            geolocation_lng REAL,
            geolocation_city TEXT,
            geolocation_state TEXT
        );"""
    },
    "order_items": {
        "description": "Line-item details for orders including price, freight value, and shipping dates. Joins with orders, products, and sellers.",
        "ddl": """CREATE TABLE order_items (
            order_id TEXT,
            order_item_id INTEGER,
            product_id TEXT,
            seller_id TEXT,
            shipping_limit_date TEXT,
            price REAL,
            freight_value REAL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
        );"""
    },
    "order_payments": {
        "description": "Payment details including payment type (credit card, boleto, voucher), installments, and value. Joins with orders.",
        "ddl": """CREATE TABLE order_payments (
            order_id TEXT,
            payment_sequential INTEGER,
            payment_type TEXT,
            payment_installments INTEGER,
            payment_value REAL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );"""
    },
    "order_reviews": {
        "description": "Customer reviews, review scores (1 to 5), and review comments/titles. Joins with orders.",
        "ddl": """CREATE TABLE order_reviews (
            review_id TEXT,
            order_id TEXT,
            review_score INTEGER,
            review_comment_title TEXT,
            review_comment_message TEXT,
            review_creation_date TEXT,
            review_answer_timestamp TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );"""
    },
    "orders": {
        "description": "Central fact table for orders. Contains order status (delivered, canceled) and timestamps. Joins with customers, order_items, order_payments, and order_reviews.",
        "ddl": """CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            order_status TEXT,
            order_purchase_timestamp TEXT,
            order_approved_at TEXT,
            order_delivered_carrier_date TEXT,
            order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );"""
    },
    "products": {
        "description": "Product catalog containing category names, dimensions, and weights. Joins with order_items and translation table.",
        "ddl": """CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_category_name TEXT,
            product_name_lenght REAL,
            product_description_lenght REAL,
            product_photos_qty REAL,
            product_weight_g REAL,
            product_length_cm REAL,
            product_height_cm REAL,
            product_width_cm REAL
        );"""
    },
    "sellers": {
        "description": "Seller demographics and locations. Joins with order_items.",
        "ddl": """CREATE TABLE sellers (
            seller_id TEXT PRIMARY KEY,
            seller_zip_code_prefix INTEGER,
            seller_city TEXT,
            seller_state TEXT
        );"""
    },
    "product_category_name_translation": {
        "description": "Lookup table mapping Portuguese product category names to English.",
        "ddl": """CREATE TABLE product_category_name_translation (
            product_category_name TEXT PRIMARY KEY,
            product_category_name_english TEXT
        );"""
    }
}

def build_sqlite_db():
    print("Building SQLite Database...")
    conn = sqlite3.connect(DB_PATH)
    # This will find the CSVs even if they are nested inside data/data/
    csv_files = glob.glob("data/**/*.csv", recursive=True)
    
    for file in csv_files:
        table_name = os.path.basename(file).replace(".csv", "").replace("olist_", "").replace("_dataset", "")
        if table_name in OLIST_METADATA:
            df = pd.read_csv(file)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Loaded {table_name}")
            
    # --- 🚀 NEW OPTIMIZATION: CREATE INDEXES FOR SPEED ---
    print("Creating Database Indexes to optimize JOINs...")
    cursor = conn.cursor()
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_order_customer ON orders(customer_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_items_product ON order_items(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_payments_order ON order_payments(order_id);",
        "CREATE INDEX IF NOT EXISTS idx_order_reviews_order ON order_reviews(order_id);"
    ]
    for idx in indexes:
        try:
            cursor.execute(idx)
        except sqlite3.OperationalError as e:
            print(f"Skipping index: {e}") # Ignores the error if a table is missing
            
    conn.commit()
    conn.close()
    print("Database built and indexed successfully!")

def build_metadata_rag():
    print("Building Semantic Metadata RAG...")
    semantic_texts = []
    payload_metadatas = []

    for table_name, data in OLIST_METADATA.items():
        semantic_texts.append(data["description"])
        payload_metadatas.append({
            "table_name": table_name,
            "schema": data["ddl"]
        })

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_texts(
        texts=semantic_texts, 
        metadatas=payload_metadatas,
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    print("Metadata RAG ready!")



if __name__ == "__main__":
    # --- AUTO-UNZIP LOGIC FOR MULTIPLE ZIPS ---
    if not os.path.exists("data"):
        os.makedirs("data") # Create the data folder if it doesn't exist

    # Extract both zip files directly into the data folder
    for zip_name in ["data1.zip", "data2.zip"]:
        if os.path.exists(zip_name):
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall("data")
            
    if not glob.glob("data/*.csv"):
        print("Please ensure the CSV files are uploaded.")
    else:
        build_sqlite_db()
        build_metadata_rag()
        print("✅ setup_db.py completed successfully!")