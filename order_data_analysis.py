import streamlit as st
import pandas as pd
import pymysql
import os
import zipfile as zp
import csv
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Capstone Project", layout="centered")

# Function to set a professional background color and add CSS for page borders and headings
def set_background_color():
    st.markdown(
        """
        <style>
        /* Background color */
        .stApp {
            background-color: #f0f2f6;
        }

        /* Page border */
        .main .block-container {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
        }

        /* Centered and italic headings */
        h1, h2, h3, h4, h5, h6 {
            text-align: center;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Home Page
def home_page():
    set_background_color()
    st.markdown("<h1 style='text-align: center; font-style: italic;'>Capstone Project</h1>", unsafe_allow_html=True)
    st.image(r"C:\Users\HP\Desktop\images.jpg", width=700)

def unzipfile():
    try:
        zip_path = r'E:\orders.csv.zip'
        output_dir = r'E:\project\Project'
        global csv_file_path

        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, 'orders_1.csv')

        with zp.ZipFile(zip_path, 'r') as zip_ref:
            with zip_ref.open('orders.csv') as file_in_zip:
                with open(csv_file_path, 'wb') as file_out:
                    file_out.write(file_in_zip.read())

        if os.path.getsize(csv_file_path) == 0:
            raise ValueError("The extracted file is empty.")

        try:
            df = pd.read_csv(csv_file_path, delimiter=',', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            df = pd.read_csv(csv_file_path, header=None)

        # Renaming columns
        df.rename(columns={
            'Order Id': 'order_id', 'Order Date': 'order_date', 'Ship Mode': 'ship_mode', 'Segment': 'segment',
            'Country': 'country', 'City': 'city', 'State': 'state', 'Postal Code': 'postal_code', 'Region': 'region',
            'Category': 'category', 'Sub Category': 'sub_category', 'Product Id': 'product_id', 'cost price': 'cost_price',
            'List Price': 'list_price', 'Quantity': 'quantity', 'Discount Percent': 'discount_percent'
        }, inplace=True)

        # Filling missing values
        df['ship_mode'] = df['ship_mode'].fillna(0)

        # Calculate discount, sale price, and profit
        df['discount'] = df['list_price'] * (df['discount_percent'] / 100)
        df['sale_price'] = (df['list_price'] - df['discount']) * df['quantity']
        df['profit'] = df['sale_price'] - (df['cost_price'] * df['quantity'])

        # Save to new file
        df.to_csv(csv_file_path, index=False)
        

    except FileNotFoundError:
        print(f"Error: The file {zip_path} does not exist.")
    except zp.BadZipFile:
        print(f"Error: The file {zip_path} is not a valid zip archive.")
    except Exception as e:
        print(f"Error: {e}")

# Function to establish database connection
def get_db_connection():
    connection = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="",
        port=3306
    )
    return connection

# Function to create database and tables
def create_database_and_tables():
    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        database_name = "project_guvi"
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        cursor.execute(f"USE {database_name}")

        table1 = """
        CREATE TABLE IF NOT EXISTS orders (
            order_id VARCHAR(50),
            order_date DATE,
            ship_mode VARCHAR(50),
            segment VARCHAR(50),
            country VARCHAR(50),
            city VARCHAR(50),
            state VARCHAR(50),
            postal_code VARCHAR(20),
            region VARCHAR(50),
            PRIMARY KEY (order_id)
        );
        """
        table2 = """
        CREATE TABLE IF NOT EXISTS products (
            order_id VARCHAR(50),
            category VARCHAR(50),
            sub_category VARCHAR(50),
            product_id VARCHAR(50),
            cost_price DECIMAL(10,2),
            list_price DECIMAL(10,2),
            quantity INT,
            discount_percent DECIMAL(5,2),
            discount DECIMAL(10,2),
            sale_price DECIMAL(10,2),
            profit DECIMAL(10,2),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
        """
        cursor.execute(table1)
        cursor.execute(table2)
        connection.commit()
    except Exception as e:
        print(f"Error creating database or tables: {e}")
    finally:
        cursor.close()
        connection.close()

# Dataset Page
def dataset_page(df_orders, df_products):
    # st.header("Dataset")
    st.markdown("<h1 style='text-align: center; font-style: italic;'> Dataset </h1>", unsafe_allow_html=True)
    st.write("Displaying first 20 records of Orders DataFrame:")
    st.dataframe(df_orders.head(20))
    st.write("Displaying first 20 records of Products DataFrame:")
    st.dataframe(df_products.head(20))

# Query Data Page
def query_data_page():
    # st.header("Query Data")
    st.markdown("<h1 style='text-align: center; font-style: italic;'>Query Data</h1>", unsafe_allow_html=True)
    query = st.text_area("Type your query here:")

    if st.button("Execute Query"):
        if query:
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute("USE project_guvi")
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [i[0] for i in cursor.description]
                result_df = pd.DataFrame(result, columns=columns)
                st.write("Query Result:")
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"Error executing query: {e}")
            finally:
                cursor.close()
                connection.close()
        else:
            st.warning("Please enter a query.")

# Questions Page
def questions_page():
    # st.header("Questions")
    st.markdown("<h1 style='text-align: center; font-style: italic;'>Questions</h1>", unsafe_allow_html=True)
    queries = {
        "Query 1": """
            SELECT p.product_id, p.category, SUM(p.sale_price * p.quantity) AS total_revenue 
            FROM products p 
            JOIN orders o ON p.order_id = o.order_id 
            GROUP BY p.product_id, p.category 
            ORDER BY total_revenue DESC 
            LIMIT 10;
        """,
        "Query 2": """
            SELECT o.city, 
            SUM(p.profit) / SUM(p.sale_price) AS profit_margin
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.city
            ORDER BY profit_margin DESC
            LIMIT 5;

        """,
        "Query 3": """
            SELECT p.category, 
            SUM(p.discount) AS total_discount
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category;

        """,
        "Query 4": """
            SELECT p.category, 
            AVG(p.sale_price) AS avg_sale_price
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category;
        """,
        "Query 5": """
            SELECT o.region, 
            AVG(p.sale_price) AS avg_sale_price
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.region
            ORDER BY avg_sale_price DESC
            LIMIT 1;

        """,
         "Query 6": """
            SELECT p.category, 
            SUM(p.profit) AS total_profit
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category;
        """,
         "Query 7": """
           SELECT o.segment, 
            SUM(p.quantity) AS total_quantity
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.segment
            ORDER BY total_quantity DESC
            LIMIT 3;

        """,
        "Query 8": """
           SELECT o.region, 
            AVG(p.discount_percent) AS avg_discount_percentage
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.region;
        """,
        "Query 9": """
           SELECT p.category, 
            SUM(p.profit) AS total_profit
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category
            ORDER BY total_profit DESC
            LIMIT 1;

        """,
        "Query 10": """
           SELECT YEAR(o.order_date) AS order_year, 
            SUM(p.sale_price * p.quantity) AS total_revenue
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY YEAR(o.order_date)
            ORDER BY order_year;


        """,
 }

    selected_queries = st.multiselect("Select queries to execute:", list(queries.keys()))
    q1_dict = {
    1: "Find top 10 highest revenue generating products",
    2: "Find the top 5 cities with the highest profit margins",
    3: "Calculate the total discount given for each category",
    4: "Find the average sale price per product category",
    5: "Find the region with the highest average sale price",
    6: "Find the total profit per category",
    7: "Identify the top 3 segments with the highest quantity of orders",
    8: "Determine the average discount percentage given per region",
    9: "Find the product category with the highest total profit",
    10: "Calculate the total revenue generated per year"
    }
    

    if st.button("Run Selected Queries"):
        for query_name in selected_queries:
            # val = q1_dict[query_name]
            query_number = int(query_name.split()[1])  # Split "Query 1" and take the second part (1)
            
            # Retrieve the description from q1_dict
            val = q1_dict.get(query_number, "Description not found")
            
            st.write(f"Executing {val}:")
            query = queries[query_name]
            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute("USE project_guvi")
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [i[0] for i in cursor.description]
                result_df = pd.DataFrame(result, columns=columns)
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"Error executing {query_name}: {e}")
            finally:
                cursor.close()
                connection.close()





def own_questions():
    # st.header("Own Questions")
    st.markdown("<h1 style='text-align: center; font-style: italic;'>Own Questions</h1>", unsafe_allow_html=True)
    queries = {
        "Query 1": """
            SELECT p.product_id, 
                   p.category, 
                   SUM(p.profit) AS total_profit
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.product_id, p.category
            ORDER BY total_profit DESC
            LIMIT 10;
        """,
        "Query 2": """
            SELECT o.state, 
                   SUM(p.sale_price * p.quantity) AS total_revenue
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.state
            ORDER BY total_revenue DESC
            LIMIT 5;
        """,
        "Query 3": """
            SELECT p.category, 
                   SUM(p.quantity) AS total_quantity_sold
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category
            ORDER BY total_quantity_sold DESC;
        """,
        "Query 4": """
            SELECT o.region, 
                   AVG(p.quantity) AS avg_quantity_sold
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.region
            ORDER BY avg_quantity_sold DESC;
        """,
        "Query 5": """
            SELECT o.city, 
                   SUM(p.discount) AS total_discount
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.city
            ORDER BY total_discount DESC
            LIMIT 1;
        """,
        "Query 6": """
            SELECT p.category, 
                   AVG(p.discount_percent) AS avg_discount
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category
            ORDER BY avg_discount DESC;
        """,
        "Query 7": """
            SELECT o.region, 
                   SUM(p.quantity) AS total_quantity_sold
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.region
            ORDER BY total_quantity_sold DESC
            LIMIT 1;
        """,
        "Query 8": """
            SELECT o.country, 
                   AVG(p.sale_price) AS avg_sale_price
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY o.country
            ORDER BY avg_sale_price DESC
            LIMIT 3;
        """,
        "Query 9": """
            SELECT p.sub_category, 
                   SUM(p.sale_price * p.quantity) AS total_revenue
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.sub_category
            ORDER BY total_revenue DESC;
        """,
        "Query 10": """
            SELECT p.category, 
                   SUM(p.discount) AS total_discount
            FROM products p
            JOIN orders o ON p.order_id = o.order_id
            GROUP BY p.category
            ORDER BY total_discount ASC
            LIMIT 1;
        """
    }

    selected_queries = st.multiselect("Select queries to execute:", list(queries.keys()))
    q2_dict = {
        1: "Find the top 10 products with the highest profit per order",
        2: "Find the top 5 states with the highest total revenue",
        3: "Calculate the total quantity sold per product category",
        4: "Find the average quantity sold per region",
        5: "Find the city with the highest total discount given",
        6: "Find the average discount per product category",
        7: "Find the region with the highest total quantity sold",
        8: "Find the top 3 countries with the highest average sale price",
        9: "Calculate the total revenue per sub-category",
        10: "Find the product category with the lowest total discount given"
    }

    if st.button("Run Selected Queries"):
        for query_name in selected_queries:
            query_number = int(query_name.split()[1])  # Extract query number
            val = q2_dict.get(query_number, "Description not found")
            st.write(f"Executing {val}:")
            query = queries[query_name]

            try:
                connection = get_db_connection()
                cursor = connection.cursor()
                cursor.execute("USE project_guvi")
                cursor.execute(query)
                result = cursor.fetchall()
                columns = [i[0] for i in cursor.description]
                result_df = pd.DataFrame(result, columns=columns)

                # Visualize the results based on the query
                if query_number == 1:
                    # Bar chart for top 10 products by profit
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="total_profit", y="product_id", data=result_df, hue="product_id", palette="viridis", legend=False)
                    plt.title("Top 10 Products by Profit")
                    plt.xlabel("Total Profit")
                    plt.ylabel("Product ID")
                    st.pyplot(plt)
                elif query_number == 2:
                    #Line chart for top 5 states by revenue
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x="state", y="total_revenue", data=result_df, marker="o")
                    plt.title("Top 5 States by Revenue")
                    plt.xlabel("State")
                    plt.ylabel("Total Revenue")
                    st.pyplot(plt)
              

                elif query_number == 3:
                    # Pie chart for total quantity sold per category
                    plt.figure(figsize=(8, 8))
                    plt.pie(result_df["total_quantity_sold"], labels=result_df["category"], autopct="%1.1f%%", startangle=140)
                    plt.title("Total Quantity Sold per Category")
                    st.pyplot(plt)
                elif query_number == 4:
                    # Bar chart for average quantity sold per region
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="region", y="avg_quantity_sold", data=result_df, hue="region", palette="magma", legend=False)
                    plt.title("Average Quantity Sold per Region")
                    plt.xlabel("Region")
                    plt.ylabel("Average Quantity Sold")
                    st.pyplot(plt)
                elif query_number == 5:
                    # Bar chart for city with the highest total discount
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="city", y="total_discount", data=result_df, hue="city", palette="viridis", legend=False)
                    plt.title("City with the Highest Total Discount")
                    plt.xlabel("City")
                    plt.ylabel("Total Discount")
                    st.pyplot(plt)

                elif query_number == 6:
                    # Line chart for average discount per category
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x="category", y="avg_discount", data=result_df, marker="o")
                    plt.title("Average Discount per Category")
                    plt.xlabel("Category")
                    plt.ylabel("Average Discount")
                    st.pyplot(plt)
                elif query_number == 7:
                    # Bar chart for region with the highest total quantity sold
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="region", y="total_quantity_sold", data=result_df, hue="region", palette="flare", legend=False)
                    plt.title("Region with the Highest Total Quantity Sold")
                    plt.xlabel("Region")
                    plt.ylabel("Total Quantity Sold")
                    st.pyplot(plt)
                elif query_number == 8:
                    # Bar chart for top 3 countries by average sale price
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x="country", y="avg_sale_price", data=result_df, hue="country", palette="crest", legend=False)
                    plt.title("Top 3 Countries by Average Sale Price")
                    plt.xlabel("Country")
                    plt.ylabel("Average Sale Price")
                    st.pyplot(plt)
                elif query_number == 9:
                    # Line chart for total revenue per sub-category
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x="sub_category", y="total_revenue", data=result_df, marker="o")
                    plt.title("Total Revenue per Sub-Category")
                    plt.xlabel("Sub-Category")
                    plt.ylabel("Total Revenue")
                    st.pyplot(plt)
                elif query_number == 10:
                    # Pie chart for product category with the lowest total discount
                    plt.figure(figsize=(8, 8))
                    plt.pie(result_df["total_discount"], labels=result_df["category"], autopct="%1.1f%%", startangle=140)
                    plt.title("Product Category with the Lowest Total Discount")
                    st.pyplot(plt)

            except Exception as e:
                st.error(f"Error executing {query_name}: {e}")
            finally:
                cursor.close()
                connection.close()


# Main Function
def main():
    unzipfile()
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        df = pd.read_csv(csv_file_path, header=None)

    df_orders = df[['order_id', 'order_date', 'ship_mode', 'segment', 'country', 'city', 'state', 'postal_code', 'region']]
    df_products = df[['order_id', 'category', 'sub_category', 'product_id', 'cost_price', 'list_price', 'quantity', 'discount_percent', 'discount', 'sale_price', 'profit']]

    
    create_database_and_tables()

    # Sidebar Navigation with Icons
    st.sidebar.markdown(
        """
        <style>
        .sidebar .stSelectbox > div > div {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
            font-style: italic;
            text-align: center;
        }
        .sidebar .stSelectbox > div > div:hover {
            background-color: #4CAF50;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar options with icons
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Dataset", "⌨️ Query Data", "❓ Questions", "❔ Own Questions"]
    )

    # Display the selected page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Dataset":
        dataset_page(df_orders, df_products)
    elif page == "⌨️ Query Data":
        query_data_page()
    elif page == "❓ Questions":
        questions_page()
    elif page == "❔ Own Questions":
        own_questions()

# Run the app
if __name__ == "__main__":
    main()
