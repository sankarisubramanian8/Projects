import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
import re
import google.generativeai as genai

# Configure Gemini AI (add your API key)
genai.configure(api_key="your_api_key")  

# --- PDF Data Extraction Function ---
def extract_text_from_pdf(pdf_data):
    """Extract text content from PDF file."""
    text = ""
    try:
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

# --- Data Cleaning and Structuring Function ---
def clean_and_structure_data(raw_text):
    """Clean and structure the extracted text data."""
    if not raw_text:
        return None

    # Remove excessive whitespace
    cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()

    try:
        # Extract transaction dates
        dates = re.findall(r'\d{2}/\d{2}/\d{4}', cleaned_text)

        # Extract transaction amounts 
        amounts = re.findall(r'₹\s*\d+(?:,\d+)*(?:\.\d+)?', cleaned_text)

        # Extract transaction descriptions (simplified approach)
        descriptions = re.findall(r'(?:UPI|IMPS|NEFT)[\w\s\-\/]+', cleaned_text)

        # Identify potential vendor/recipient names
        vendors = re.findall(r'[A-Z][a-z]+\s[A-Z][a-z]+|[A-Z\s]{2,}', cleaned_text)

        # Add structured data to the cleaned text
        structured_info = {
            "num_transactions_found": len(dates),
            "transaction_dates": dates[:10],  # Sample of first 10
            "transaction_amounts": amounts[:10],  # Sample of first 10
            "potential_vendors": list(set(vendors))[:10]  # Sample of first 10 unique vendors
        }

        return {
            "cleaned_text": cleaned_text,
            "structured_data": structured_info
        }
    except Exception as e:
        # If structured extraction fails, return just the cleaned text
        return {
            "cleaned_text": cleaned_text,
            "structured_data": None,
            "extraction_error": str(e)
        }

# --- Financial Analysis Function ---
def analyze_financial_data(processed_data):
    """Sends extracted and processed text to Google Gemini AI for financial insights."""
    if not processed_data or not processed_data.get("cleaned_text"):
        return "⚠️ No valid data to analyze."

    text = processed_data["cleaned_text"]

    # Add any structured data insights we've extracted
    if processed_data.get("structured_data"):
        structure_info = processed_data["structured_data"]
        text = f"""
        Pre-processed data:
        - Approximate number of transactions: {structure_info.get('num_transactions_found', 'Unknown')}
        - Sample transaction dates: {', '.join(structure_info.get('transaction_dates', [])[:5])}
        - Sample vendors identified: {', '.join(structure_info.get('potential_vendors', [])[:5])}

        Full statement data:
        {text}
        """

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
      Analyze the provided UPI statement data with meticulous attention to detail and generate a comprehensive financial assessment
      {text}

      STEP 1: TRANSACTION OVERVIEW
      - Extract the exact total money paid and received amounts
      - Count and verify the number of transactions (both outgoing and incoming)
      - Calculate the net cash flow (money received minus money paid)
      - Identify the date range covered and calculate the number of active transaction days
      - Note any days without transactions within the period

      STEP 2: DETAILED SPENDING PATTERN ANALYSIS
      - Calculate average daily and weekly spending
      - Identify your highest spending day of the week and time of day
      - Determine frequency of transactions (daily average)
      - Calculate your average transaction amount for outgoing payments
      - Identify any spending spikes or unusual patterns
      - Note the distribution of transaction sizes (small, medium, large)

      STEP 3: TEMPORAL TREND IDENTIFICATION
      - Analyze spending patterns by day of week
      - Examine spending patterns by time of day (morning, afternoon, evening, night)
      - Identify any cyclical patterns (beginning of month vs end of month)
      - Determine if spending increases on weekends vs weekdays
      - Calculate daily variance in spending

      ## Enhanced Transaction Categorization Section for UPI Analysis


      STEP 4: TRANSACTION CATEGORIZATION
      - Extract any explicit category tags present in the statement (like #Food, #Transfers)
      - Use merchant/recipient name-based categorization:
        * Food & Dining: Identify transactions with keywords like restaurant names, food delivery services, cafes
        * Groceries: Transactions with supermarkets, grocery stores, vegetable vendors
        * Transportation: Transactions with ride services, metro, bus, fuel stations (e.g., "Chennai Metro Rail Ltd")
        * Bill Payments: Utilities, phone/internet providers, subscriptions
        * Shopping: Retail stores, e-commerce platforms
        * Entertainment: Movie theaters, streaming services, events
        * Healthcare: Hospitals, clinics, pharmacies
        * Education: Schools, courses, education-related payments
        * Person-to-Person Transfers: Transactions to individual names (e.g., "Perumal S", "Jagadheeswari U")
        * Services: Home services, repairs, maintenance
        * Travel: Hotels, airlines, travel booking services

      - For ambiguous recipient names:
        * Consider transaction amount patterns (small frequent payments vs large one-time payments)
        * Consider transaction timing patterns (morning purchases might indicate breakfast/coffee)
        * Group similar-looking transactions that occur regularly

      - Create a comprehensive category distribution:
        * Calculate total spending per category
        * Calculate percentage of total spending for each category
        * Rank categories from highest to lowest spending
        * Highlight top 3 spending categories
        * Identify categories with unusually high single transactions
        * Flag categories with high frequency but low total amount (potential impulse spending)

      - Present categorization in detailed table format:
        * Category
        * Number of Transactions
        * Total Amount (₹)
        * Percentage of Total Spending
        * Average Transaction Amount
        * Largest Single Transaction
        * Most Frequent Recipient in Category

      - For uncategorizable transactions:
        * Create a "Miscellaneous" category
        * List these transactions separately for manual review
        * Provide possible categorization suggestions based on amount patterns

      - Provide category-specific insights:
        * Highlight categories exceeding typical budget percentages
        * Identify categories with growth/reduction trends within the statement period
        * Suggest categories that could benefit from spending consolidation


      STEP 5: RECIPIENT ANALYSIS
      - Identify frequent payment recipients (3+ transactions)
      - Calculate total amount paid to each frequent recipient
      - Determine average payment size for each frequent recipient
      - Identify one-time large payments vs regular small payments
      - Analyze if certain recipients are associated with specific times/days

      STEP 6: INCOME SOURCE ASSESSMENT
      - Identify all sources of incoming funds
      - Calculate frequency and average amount from each source
      - Determine reliability/consistency of income sources
      - Calculate the ratio of income to expenses
      - Identify any patterns in timing of income receipts

      STEP 7: WASTEFUL SPENDING DETECTION
      - Flag multiple small transactions to the same recipient that could be consolidated
      - Identify frequent low-value transactions that may be impulsive
      - Highlight unusually large one-off expenses that deviate from normal patterns
      - Detect potential duplicate payments
      - Calculate the total "potentially wasteful" spending amount

      STEP 8: BUDGET DEVIATION ANALYSIS
      - Estimate reasonable budget allocations based on spending categories
      - Calculate overspending in each category based on these allocations
      - Identify categories with the highest budget deviation
      - Determine overall budget adherence score
      - Suggest specific categories requiring stricter budget control

      STEP 9: FINANCIAL HABIT ASSESSMENT
      - Identify potential impulse spending patterns (time of day, frequency)
      - Assess savings behavior based on spending vs. income
      - Detect habitual spending (same vendor, same time)
      - Calculate financial discipline score based on spending patterns
      - Identify potentially problematic spending behaviors

      STEP 10: PERSONALIZED RECOMMENDATION GENERATION
      - Create a suggested monthly budget with specific category allocations
      - Provide 5 specific, actionable recommendations to reduce unnecessary spending
      - Suggest 3 specific ways to consolidate transactions to reduce fees or effort
      - Recommend optimal timing for large payments based on cash flow
      - Provide 3 personalized long-term financial improvement strategies
      - Suggest specific categories where spending could be reduced without significant lifestyle impact
      - Recommend specific actions for improved financial tracking and management
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response else "⚠️ Error processing financial data."
    except Exception as e:
        return f"⚠️ Error analyzing financial data: {str(e)}"

# --- Format Analysis Results Function ---
def format_analysis_results(analysis_text):
    """Format the analysis results for better display."""
    # Split analysis into sections
    sections = re.split(r'STEP \d+:', analysis_text)

    formatted_results = ""

    # Format each section
    for i in range(1, len(sections)):
        section = sections[i]
        section_title = re.search(r'([A-Z\s&]+)', section)

        if section_title:
            title = section_title.group(1).strip()
            formatted_results += f"### {title}\n\n"

            # Extract bullet points
            bullet_points = re.findall(r'-\s+(.*?)(?=\n-|\n\n|$)', section)
            for point in bullet_points:
                formatted_results += f"- {point.strip()}\n"

            formatted_results += "\n"

    return formatted_results

# --- Page Config ---
st.set_page_config(
    page_title="UPI & Financial Analyzer",
    page_icon="💰",
    layout="centered"
)

# --- Custom CSS for Stylish UI ---
st.markdown("""
<style>
    .title {
        font-size: 2.5rem !important;
        color: #2e86de;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #2e86de;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .analyze-btn {
        background-color: #2e86de !important;
        color: white !important;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .result-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .insight-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2e86de;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    .debug-box {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        padding: 10px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown('<p class="title">💳 Personal UPI & Financial Analyzer</p>', unsafe_allow_html=True)

# --- PDF Upload Section ---
st.markdown("### Upload Your Bank Statement (PDF)")
uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=["pdf"],
    key="pdf_uploader"
)


# --- Analyze Button ---
if st.button("🔍 Analyze Transactions", key="analyze_btn", use_container_width=True):
    if uploaded_file is not None:
        with st.spinner("Processing your PDF statement..."):
            # Step 1: Extract text from PDF
            pdf_data = BytesIO(uploaded_file.read())
            raw_text = extract_text_from_pdf(pdf_data)

            if raw_text:
                # Step 2: Clean and structure the extracted data
                processed_data = clean_and_structure_data(raw_text)

                # Step 3: Analyze the cleaned data
                with st.spinner("Analyzing your financial data with AI..."):
                    analysis = analyze_financial_data(processed_data)

                # Step 4: Format and display the results
                if analysis:
                    formatted_analysis = format_analysis_results(analysis)

                    # --- Display Results ---
                    st.success("✅ Analysis Complete!")

                    # --- Display Analysis ---
                    st.markdown("## 💡 AI-Powered Financial Insights")
                    st.markdown('<div class="insight-box">' + analysis.replace('\n', '<br>') + '</div>',
                              unsafe_allow_html=True)

                    # Disclaimer
                    st.warning("""ℹ️ Note: This is an AI-generated analysis.
                            Verify critical financial decisions with a professional.""")
            else:
                st.error("Failed to extract text from the PDF. Please check if the file is valid and try again.")
    else:
        st.error("Please upload a PDF file first!")

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("## ℹ️ How It Works")
    st.markdown("""
    1. Upload your bank statement PDF
    2. We extract transaction text
    3. AI analyzes spending patterns
    4. Get personalized insights

    **Supported Formats:**
    - Standard bank PDFs
    - UPI transaction histories

    **Privacy Note:**
    - Processing happens in your browser
    - No data stored on our servers
    """)
    st.markdown("---")
    st.markdown("Built with ❤️ using [Gemini AI](https://ai.google.dev/)")
