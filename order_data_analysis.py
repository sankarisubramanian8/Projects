import streamlit as st
import pandas as pd
import pymysql
import os
import zipfile as zp
import csv


st.set_page_config(page_title="Capstone Project", layout="centered")


def home_page():
    st.markdown("<h1 style='text-align: center;'>Capstone Project</h1>", unsafe_allow_html=True)
    st.image(r"C:\Users\HP\Desktop\images.jpg",width = 700)  

def unzipfile():
    try:
        zip_path = r'E:\orders.csv.zip'


        output_dir = r'E:\project\Project' 
        global csv_file_path

        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, 'orders_1.csv')
    

        with zp.ZipFile(zip_path, 'r') as zip_ref:
            print("Files in the zip archive:", zip_ref.namelist())
        
            
            with zip_ref.open('orders.csv') as file_in_zip:
                with open(csv_file_path, 'wb') as file_out:
                    file_out.write(file_in_zip.read())

            if os.path.getsize(csv_file_path) == 0:
                raise ValueError("The extracted file is empty.")
 
        try:
            df = pd.read_csv(
                    csv_file_path,
                    delimiter=',',  
                    quoting=csv.QUOTE_MINIMAL,  
                    on_bad_lines='skip'  
                )
        except pd.errors.ParserError as e:
            
            print(f"ParserError: {e}")
            df = pd.read_csv(csv_file_path, header=None)
            print("CSV file read in fallback mode (no header).")


        #Renaming column

        df.rename(columns={'Order Id': 'order_id', 'Order Date': 'order_date','Ship Mode': 'ship_mode','Segment': 'segment',
                        'Country': 'country','City': 'city','State': 'state',
                        'Postal Code': 'postal_code','Region': 'region','Category': 'category','Sub Category': 'sub_category',
                        'Product Id': 'product_id','cost price': 'cost_price','List Price': 'list_price',
                            'Quantity': 'quantity','Discount Percent':'discount_percent'}, inplace=True)
        
        
        # filling 0 values in missing place
        df['ship_mode'] = df['ship_mode'].fillna(0)
        print(df['ship_mode'][df['ship_mode'].isnull()])

        #step 1: calculate discount
        df['discount'] = df['list_price'] * (df['discount_percent'] / 100)
        # Step 2: Calculate the sale price
        df['sale_price'] = (df['list_price'] - df['discount']) * df['quantity']

        # Step 3: Calculate the profit
        df['profit'] = df['sale_price'] - (df['cost_price'] * df['quantity'])
        
        #saving to new file
        df.to_csv(csv_file_path, index=False)
        print("unzip done successfully")

    except FileNotFoundError:
        print(f"Error: The file {zip_path} does not exist.")
    except zp.BadZipFile:
        print(f"Error: The file {zip_path} is not a valid zip archive.")
    except Exception as e:
        print(f"Error: {e}")


# Dataset Page
def dataset_page(df_orders, df_products):
    
    st.header("Dataset")

    # Display the first 20 records of DataFrame 1
    st.write("Displaying first 20 records of DataFrame 1:")
    
    st.dataframe(df_orders.head(20))

    # Display the first 20 records of DataFrame 2
    st.write("Displaying first 20 records of DataFrame 2:")

    st.dataframe(df_products.head(20))
    # 
# Part1 Page
def part1_page():
    st.header("Part 1: Query Database")
    query = st.text_area("Type your query here:")

    if st.button("Execute Query"):
        if query:
            try:
                
                connection = pymysql.connect(host="127.0.0.1",
                user="root",
                password="",
                port=3306)  
                cursor = connection.cursor()

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

                cursor.execute(query)
                result = cursor.fetchall()
                columns = [i[0] for i in cursor.description] 
                result_df = pd.DataFrame(result, columns=columns)
                st.write("Query Result:")
                st.dataframe(result_df)
                cursor.close()
                connection.close()
            except Exception as e:
                st.error(f"Error executing query: {e}")
        else:
            st.warning("Please enter a query.")


def main():
    unzipfile()
    try:
        df = pd.read_csv(csv_file_path, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        df = pd.read_csv(csv_file_path, header=None)

    df_orders = df[['order_id', 'order_date', 'ship_mode', 'segment', 'country', 'city', 'state', 'postal_code', 'region']]
    df_products = df[['order_id', 'category', 'sub_category', 'product_id', 'cost_price', 'list_price', 'quantity', 'discount_percent', 'discount', 'sale_price', 'profit']]

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Home", "Dataset", "Query_Data"])

    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Dataset":
        dataset_page(df_orders, df_products)
    elif page == "Query_Data":
        part1_page()

# Run the app 
if __name__ == "__main__":
    main()